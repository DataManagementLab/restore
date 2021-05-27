import logging
from copy import copy
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from numba import njit, prange

from join_completion.models.completion_model import CompletionSetup, CompletionModel
from schema_setup.schema.dataset import Dataset
from schema_setup.schema.schema_utils import join_tables, extend_by_rels
from ssar import common
from ssar.common import Discretize
from ssar.train_model import train_autoregressive

logger = logging.getLogger(__name__)


@njit()
def replace_idxs(data, j, distinct_vals):
    for i in prange(data.shape[0]):
        idx = int(data[i, j])
        if 0 < idx < len(distinct_vals):
            data[i, j] = distinct_vals[idx]
        else:
            data[i, j] = np.nan
    return data


class FlatARCompletionModel(CompletionModel):
    def __init__(self, model_directory, r=None, inverse=False, params=None):
        CompletionModel.__init__(self, model_directory)

        self.merged = False
        self.learned_attributes = None
        self.columns = None

        # hyperparams
        self.params = params

        if r is not None:

            cs = CompletionSetup(set(), r, set(), inverse)

            # scenario: completion of order->orderline
            # additional helps if we join customer
            self.expand_evidence(cs.evidence_relationships, cs.evidence_table, r)

            # scenario: orderline->orders should be completed
            # helps if model also learns customers (orders with unknown orderline but known customers can be matched
            # easier) - we no longer do matching so this can be simplified
            # if len(r.pks_without_fk) > 0:
            #     self.expand_completion(cs.completion_relationships, cs.completion_table, r)

            self.completion_relationships = {cs}

            # later on the AR model has to be able to perform all the completions. This restricts the variable ordering
            # for the AR model (evidence relationships before r before completion_relationships). We encode this a
            # directed graph.
            self.r_graph = nx.DiGraph()
            self.r_graph.add_node(r)
            for r_e in cs.evidence_relationships:
                self.r_graph.add_edge(r_e, r)
                for r_c in cs.completion_relationships:
                    self.r_graph.add_edge(r_e, r_c)
            for r_c in cs.completion_relationships:
                self.r_graph.add_edge(r, r_c)
            assert nx.is_directed_acyclic_graph(self.r_graph)

    def map_to_validation_schema(self, t_mapping, r_mapping, a_mapping):
        mapped_model = copy(self)

        mapped_model.completion_relationships = []

        for cs in self.completion_relationships:
            mapped_model.completion_relationships.append(
                CompletionSetup({r_mapping[r] for r in cs.evidence_relationships}, r_mapping[cs.r],
                                {r_mapping[r] for r in cs.completion_relationships}, cs.inverse))

        mapped_model.learned_attributes = [a_mapping[a] for a in self.learned_attributes]
        return mapped_model

    def return_parameters(self):
        params = CompletionModel.return_parameters(self)
        if self.params is not None:
            params.update(self.params)

        return params

    @property
    def relationships(self):
        relationships = set()
        for cs in self.completion_relationships:
            relationships.add(cs.r)
            relationships.update(cs.evidence_relationships)
            relationships.update(cs.completion_relationships)
        return relationships

    @property
    def evidence_tables(self):
        tables = set()
        for cs in self.completion_relationships:
            tables.update(cs.evidence_tables)
        return tables

    @property
    def tables(self):
        tables = set()
        for cs in self.completion_relationships:
            tables.update(cs.tables)
        return tables

    def __str__(self):
        str_rep = 'FlatARModel(relations=' + ','.join([str(t) for t in self.table_ordering()]) + ')'
        for cr in self.completion_relationships:
            str_rep += ('\n\t' + str(cr))
        return str_rep

    def table_ordering(self):
        return nx.topological_sort(self.r_graph)

    def expand_evidence(self, evidence_r, table, completion_r):
        for r in table.outgoing_relationships:
            if r == completion_r or r in evidence_r:
                continue
            evidence_r.add(r)
            self.expand_evidence(evidence_r, r.outgoing_table, completion_r)

    def train(self):
        _, table = self._training_data()

        self.params['ignore_sets'] = True
        acc, training_time, model = train_autoregressive(table, self.model_directory, self.model_name, **self.params)

        self.model = model
        return training_time, acc

    def _training_data(self):
        join_relationships = list(self.table_ordering())
        full_join_dataset = join_tables(join_relationships, relationships_ordered=True, incomplete_join=True,
                                        how='left')

        self.learned_attributes = [a for a in full_join_dataset.attributes if not (a.is_pk or a.is_fk)]
        join_dataset = full_join_dataset.project(self.learned_attributes)

        table = common.CsvTable(None, join_dataset.df_rows, join_dataset.df_rows.columns, {})
        self.columns = table.columns
        assert len(self.columns) == len(self.learned_attributes)

        return full_join_dataset, table

    def transform_to_evidence(self, data, columns=None):
        if columns is None:
            columns = self.columns

        for i in range(data.shape[1]):
            data[:, i] = Discretize(columns[i], data=data[:, i])

        return data

    def transform_back(self, data, col_idxs=None):
        for i in range(data.shape[1]):
            col_idx = i
            if col_idxs is not None:
                col_idx = col_idxs[i]
            col = self.columns[col_idx]
            assert col.all_distinct_values[0] == -1

            data = replace_idxs(data, i, col.all_distinct_values)

        return data

    def find_cs_r(self, r, inverse=True):
        if inverse is None:
            potential_completion_setups = [cs for cs in self.completion_relationships if cs.r == r]
        else:
            potential_completion_setups = [cs for cs in self.completion_relationships if
                                           cs.r == r and cs.inverse == inverse]
        assert len(potential_completion_setups) == 1
        return potential_completion_setups[0]

    def complete_1_n(self, current_join, next_table, r, max_samples=None, suppress_nan=True,
                     virtual_ids=True, keep_ids=True, percentile=None, percentile_attributes=None,
                     predictability_score=None):
        # find corresponding completion setup
        cs_r = self.find_cs_r(r, inverse=False)
        logger.info(f"Completing 1:n {cs_r} ({len(current_join.df_rows)} rows)")

        tf_attribute = cs_r.r.tf_attribute
        tf_idx, tf_weights = self.tf_weights(cs_r.r, cardinality_correction=True)

        # find evidence attributes we need
        original_current_join = current_join
        current_join = self.extend_required_evidence(cs_r, current_join)
        evidence, completed_attrs, completed_attr_max_idx = \
            self.project_evidence(cs_r, current_join)
        hierarchy_batches = self.transform_hierarchy_evidence(current_join)

        # first predict the tuple factors for the current_join
        # sample weighted by tf to have cardinality correction
        start_t = perf_counter()
        # forget about previously sampled tfs
        # but remember other columns we might have sampled earlier
        remember_evidence = evidence[:, tf_idx + 1:]
        evidence = evidence[:, :tf_idx]

        sample = self.model.sample(evidence=evidence, pos_weights=tf_weights, end_idx=tf_idx,
                                   device=self.params['device'], bs=self.params['inference_bs'],
                                   hierarchy_batches=hierarchy_batches)
        logger.info(f"\tSampled {len(sample)} tfs in {perf_counter() - start_t:.2f}s")
        tfs = sample[:, tf_idx]

        # some tuple factors are known - i.e., we can discard the predictions here
        tfs_known = current_join.df_rows[tf_attribute.full_name]
        ev_tfs_known = Discretize(self.columns[tf_idx], data=tfs_known)
        # unknown values are transformed to nan
        known_idx = ev_tfs_known > 0
        tfs[known_idx] = ev_tfs_known[known_idx]

        assert np.all(tfs >= 0)
        evidence = np.concatenate((evidence, tfs.reshape(-1, 1)), axis=1)
        if remember_evidence.shape[1] > 0:
            evidence = np.concatenate((evidence, remember_evidence), axis=1)

        # reduce expected tf by current tf
        expected_tfs = self.transform_back(tfs.reshape(-1, 1), col_idxs=[tf_idx]).reshape(-1)
        assert not np.any(np.isnan(expected_tfs))

        expected_avg_tf = expected_tfs.mean()
        logger.info(f"expected_avg_tf: {expected_avg_tf}")
        additional_tuples = expected_tfs - current_join.df_rows['current_tf']
        additional_tuples = np.clip(additional_tuples, 0, np.inf)
        logger.info(f"Having {additional_tuples.mean()} additional tuples on average")
        current_mean = current_join.df_rows['current_tf'].mean()
        if additional_tuples.mean() > 0 and expected_avg_tf > current_mean:
            logger.info(f"Normalizing tuple factors expected_avg_tf: {expected_avg_tf}, current_join: {current_mean}")
            additional_tuples *= (expected_avg_tf - current_mean) / additional_tuples.mean()
            assert np.isclose(current_join.df_rows['current_tf'].mean() + additional_tuples.mean(), expected_avg_tf)

        # edge case
        additional_tuples = np.clip(additional_tuples, 0, np.inf)

        logger.info(f"Having {additional_tuples.mean()} additional tuples on average")
        evidence = np.repeat(evidence, additional_tuples.astype(int), axis=0)
        hierarchy_batches = self.reshape_hierarchy_evidence(hierarchy_batches, additional_tuples=additional_tuples)

        sample_idx, sample = self.sample_remaining(completed_attr_max_idx, completed_attrs, evidence, max_samples,
                                                   suppress_nan, hierarchy_batches=hierarchy_batches,
                                                   percentile=percentile, percentile_attributes=percentile_attributes,
                                                   predictability_score=predictability_score)

        synthesized = self.make_dataset(self.learned_attributes[:sample.shape[1]], sample,
                                        virtual_ids=virtual_ids, next_table=next_table,
                                        original_current_join=original_current_join, keep_ids=keep_ids,
                                        additional_tuples=additional_tuples, sample_idx=sample_idx)
        return synthesized

    def reshape_hierarchy_evidence(self, hierarchy_batches, additional_tuples=None, sample_idx=None):
        return None

    def transform_hierarchy_evidence(self, current_join):
        return None

    def make_dataset(self, current_attributes, sample, virtual_ids=False, next_table=None,
                     original_current_join=None, keep_ids=False, additional_tuples=None, sample_idx=None):
        assert len(current_attributes) == sample.shape[1]

        # project only to attributes that were originally in the join or are projected
        final_attributes = [a for a in self.learned_attributes if
                            a in original_current_join.attributes or a in next_table.attributes]
        # project sample to these attributes
        assert set(final_attributes).issubset(current_attributes)
        proj = [i for i, a in enumerate(current_attributes) if a in final_attributes]
        sample = sample[:, proj]
        assert len(final_attributes) == sample.shape[1]

        # keep already existing primary keys
        for i, a in enumerate(original_current_join.attributes):
            # if a.is_pk and a not in final_attributes:
            if a not in final_attributes:
                a_values = original_current_join.df_rows.values[:, i]
                if a.is_pk and not keep_ids:
                    continue
                if np.all(np.isnan(a_values)) or a.full_name == 'current_tf':
                    continue
                if additional_tuples is not None:
                    a_values = np.repeat(a_values, additional_tuples.astype(int))
                if sample_idx is not None:
                    a_values = a_values[sample_idx]
                sample = np.concatenate((a_values.reshape(-1, 1), sample), axis=1)
                final_attributes = [a] + final_attributes

        # whether artificial primary keys should be assigned for generated table (might be useful in later steps of
        #   the join)
        if virtual_ids:
            # simply ascending negative integers as pks
            virtual_id_col = np.reshape(-np.arange(1, len(sample) + 1), (-1, 1))

            # find corresponding attribute
            pk_candidates = [a for a in next_table.attributes if a.is_pk]
            assert len(pk_candidates) == 1
            pk_attr = pk_candidates[0]

            # nothing to do otherwise
            if pk_attr not in final_attributes:
                # concatenate
                sample = np.concatenate((virtual_id_col, sample), axis=1)
                final_attributes = [pk_attr] + final_attributes

        synthesized_df = pd.DataFrame(sample, columns=[a.full_name for a in final_attributes])
        return Dataset(synthesized_df, final_attributes)

    def sample_remaining(self, completed_attr_max_idx, completed_attrs, evidence, max_samples, suppress_nan,
                         hierarchy_batches=None, pos_weights=None, pos_weights_idx=None, percentile=None,
                         percentile_attributes=None, predictability_score=None):
        idx = None
        if max_samples is not None:
            idx = np.random.randint(evidence.shape[0], size=max_samples)
            evidence = evidence[idx]
            hierarchy_batches = self.reshape_hierarchy_evidence(hierarchy_batches, sample_idx=idx)

        percentile_idxs = None
        percentile_inverted = None
        percentile_idxs_values = None
        if percentile is not None:
            percentile_idxs = []
            percentile_idxs_values = []
            percentile_inverted = []
            attribute_names = [a.full_name for a in self.learned_attributes]
            for percentile_a in percentile_attributes:
                if percentile_a.attribute_name in attribute_names:
                    percentile_idxs.append(attribute_names.index(percentile_a.attribute_name))
                    percentile_idxs_values.append(percentile_a.attribute_value)
                    percentile_inverted.append(percentile_a.inverted)

        if len(completed_attrs) > 0:
            start_t = perf_counter()
            logger.info(f"\tSampling {len(evidence)} evidence tuples")
            sample = self.model.sample(evidence=evidence, device=self.params['device'], bs=self.params['inference_bs'],
                                       end_idx=completed_attr_max_idx, suppress_nan=suppress_nan,
                                       hierarchy_batches=hierarchy_batches, pos_weights=pos_weights,
                                       pos_weights_idx=pos_weights_idx, percentile_idxs=percentile_idxs,
                                       percentile_inverted=percentile_inverted, percentile=percentile,
                                       percentile_idxs_values=percentile_idxs_values,
                                       predictability_score=predictability_score)
            logger.info(f"\tSampled {len(sample)} evidence tuples in {perf_counter() - start_t:.2f}s")
        else:
            sample = evidence
        # transform this entire sample back, s.t. it does not use internal transform_tensors of AR model
        start_t = perf_counter()
        sample = sample[:, :completed_attr_max_idx + 1]
        sample = self.transform_back(sample)
        logger.info(f"\tTransforming back {len(sample)} tuples took {perf_counter() - start_t:.2f}s")
        return idx, sample

    def project_evidence(self, cs_r, current_join, exclude=None):
        ev_attributes = cs_r.project_evidence_attributes(self.learned_attributes, exclude=exclude)
        projected_current_join = current_join.project(ev_attributes)
        evidence = projected_current_join.df_rows.values
        evidence = self.transform_to_evidence(evidence)

        # find attributes we want to predict
        completed_attrs = [idx for idx, a in enumerate(self.learned_attributes) if
                           a in set(cs_r.completion_table.attributes)]
        if len(completed_attrs) > 0:
            completed_attr_max_idx = max(completed_attrs)
        else:
            completed_attr_max_idx = evidence.shape[1]

        return evidence, completed_attrs, completed_attr_max_idx

    @property
    def unique_set_pk(self):
        set_pk = None
        inverse_completion_relationships = [cs_r for cs_r in self.completion_relationships if not cs_r.inverse]
        if len(inverse_completion_relationships) == 1:
            set_pk = inverse_completion_relationships[0].evidence_table.primary_key[0]
        return set_pk

    def complete_n_1(self, current_join, next_table, r, max_samples=None, suppress_nan=True, virtual_ids=True,
                     keep_ids=True, percentile=None, percentile_attributes=None,
                     predictability_score=None):
        # find corresponding completion setup
        cs_r = self.find_cs_r(r, inverse=True)
        logger.info(f"Completing n:1 {cs_r} ({len(current_join.df_rows)} rows)")

        # find evidence attributes we need
        original_current_join = current_join
        current_join = self.extend_required_evidence(cs_r, current_join)
        evidence, completed_attrs, completed_attr_max_idx = self.project_evidence(cs_r, current_join)
        # one set once = False?
        hierarchy_batches = self.transform_hierarchy_evidence(current_join)

        # make sure we do not sample zero tfs (would not appear in the join)
        tf_idx, tf_weights = self.tf_weights(cs_r.r, ge_zero=True)
        sample_idx, sample = self.sample_remaining(completed_attr_max_idx, completed_attrs, evidence, max_samples,
                                                   suppress_nan, hierarchy_batches=hierarchy_batches,
                                                   pos_weights=tf_weights,
                                                   pos_weights_idx=tf_idx,
                                                   percentile=percentile, percentile_attributes=percentile_attributes,
                                                   predictability_score=predictability_score)
        synthesized = self.make_dataset(self.learned_attributes[:completed_attr_max_idx + 1], sample,
                                        virtual_ids=virtual_ids, next_table=next_table,
                                        original_current_join=original_current_join, keep_ids=keep_ids,
                                        additional_tuples=None, sample_idx=sample_idx)
        return synthesized

    def extend_required_evidence(self, cs_r, current_join):
        return extend_by_rels(current_join, cs_r.evidence_relationships)

    def project_single_column_evidence(self, attribute, current_join):
        ev_attributes = []
        for a in self.learned_attributes:
            if a == attribute:
                break
            ev_attributes.append(a)

        projected_current_join = current_join.project(ev_attributes)
        evidence = projected_current_join.df_rows.values
        evidence = self.transform_to_evidence(evidence)

        # find attributes we want to predict
        completed_attrs = [idx for idx, a in enumerate(self.learned_attributes) if a == attribute]
        completed_attr_max_idx = max(completed_attrs)

        return evidence, completed_attrs, completed_attr_max_idx

    def predict_tf(self, r, completion_table, current_join, suppress_nan=True, predict_idx=None, ge_zero=True,
                   percentile=None, percentile_attributes=None, predictability_score=None):
        if predict_idx is not None:
            if len(predict_idx[predict_idx]) == 0:
                return []
            # copy to avoid affecting the other parts
            current_join = copy(current_join)
            current_join.df_rows = current_join.df_rows[predict_idx]
            current_join.weights = current_join.weights[predict_idx]

        _, pos_weights = self.tf_weights(r, ge_zero=True)

        cs_r = self.find_cs_r(r, inverse=None)
        current_join = self.extend_required_evidence(cs_r, current_join)
        evidence, completed_attrs, completed_attr_max_idx = self.project_single_column_evidence(r.tf_attribute,
                                                                                                current_join)
        hierarchy_batches = self.transform_hierarchy_evidence(current_join)
        # note that no pos_weights are required here since we already executed the join (e.g., if we see a tuple factor
        #   of 3, this tuple will really appear three times in the join)
        _, sample = self.sample_remaining(completed_attr_max_idx, completed_attrs, evidence, None, suppress_nan,
                                          hierarchy_batches=hierarchy_batches, pos_weights=pos_weights,
                                          percentile=percentile, percentile_attributes=percentile_attributes,
                                          predictability_score=predictability_score
                                          )
        tfs = sample[:, completed_attr_max_idx]

        if ge_zero:
            assert np.all(tfs > 0)

        return tfs

    def tf_weights(self, r, ge_zero=False, cardinality_correction=False):
        pos_weights = None
        tf_idx = self.learned_attributes.index(r.tf_attribute)
        if ge_zero:
            pos_weights = [0 if tf < 1 else 1 for tf in self.columns[tf_idx].all_distinct_values]

        if cardinality_correction:
            assert not ge_zero

            def map_tf_prob(tf):
                if tf < 0:
                    return 0
                if tf >= 1:
                    return 1 / tf
                return 1

            pos_weights = [map_tf_prob(tf) for tf in self.columns[tf_idx].all_distinct_values]

        return tf_idx, pos_weights
