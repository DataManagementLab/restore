import logging
from copy import copy
from enum import Enum

import numpy as np

from evaluation.relative_error.rerr_reduction import evaluate_rerr_reduction, rel_err_reduction, mae
from join_completion.query_compilation.operators.ann_replacement import ANN_Replacement
from join_completion.query_compilation.operators.join import JoinRelationship
from join_completion.query_compilation.operators.load import LoadCompleteTable
from join_completion.query_compilation.operators.projection import ProjectRequestedJoin
from join_completion.query_compilation.operators.top_path_union import TopPathUnion, UnionStrategy
from join_completion.query_compilation.query import Query
from schema_setup.incomplete_schema_setup.incomplete_schema_generation import _incomplete_schema, \
    derive_incomplete_schema
from schema_setup.incomplete_schema_setup.removal_method import RemovalMethod
from schema_setup.schema.dataset import Dataset
from schema_setup.schema.schema_utils import custom_bfs, relationship_order

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    NONE = 'none'
    ARTIFICIAL_BIAS = 'artificial_bias'
    ARTIFICIAL_BIAS_MAE = 'artificial_mae'
    UNIFORM_MAE = 'uniform_mae'
    COMBINED = 'combined'

    def __str__(self):
        return self.value


class PercentileAttribute:
    def __init__(self, attribute_name, attribute_value, inverted=False):
        self.attribute_name = attribute_name
        self.attribute_value = attribute_value
        self.inverted = inverted


class PredictabilityScore(Enum):
    PRIOR_VAL = 'prior_val'
    KL_DIV_PRIOR = 'kl_div_prior'

    def __str__(self):
        return self.value


def incomplete_join_plan(schema, model_families, query, selection_strategy, max_samples=None, fully_synthetic=True,
                         suppress_nan=False, validation_removal_attr=None, validation_removal_attr_values=None,
                         validation_removal_method=None, validation_tuple_removal_table=None,
                         validation_tuple_removal_keep_rate=None, validation_removal_attr_bias=None,
                         validation_tf_keep_rate=None, ann_batch_size=10000, ann_neighbors_considered=100000,
                         fixed_completion_path=None, top_path_union_strategy=UnionStrategy.COMBINE,
                         force_no_baseline=False, force_path_selection=False, percentile=None,
                         percentile_attributes=None, predictability_score=None):
    """
    Returns the most naive query plan for a query. First finds anchor table (which is complete) using bfs in schema if
        there is no complete table in the target_relationships. No matching is done at all, very naive.
    """

    requested_tables = find_requested_tables(query)

    def find_completion_paths(table=None, r_seq=None, paths=None, model_families=None, target_relationships=None,
                              **kwargs):
        if table.complete:
            paths.append(CompletionPath(schema, table, r_seq, model_families, requested_tables, target_relationships))
            return True
        return False

    paths = []
    custom_bfs(requested_tables, process_step=find_completion_paths, paths=paths, model_families=model_families,
               target_relationships=query.target_relationships)

    assert len(paths) > 0, "No path found"

    if fixed_completion_path is not None:
        fixed_completion_path = {schema.table_dict[t] for t in fixed_completion_path}
        paths = [path for path in paths if set(fixed_completion_path).union(requested_tables) == set(path.tables)]

    assert len(paths) > 0, f"No path found due to fixed_completion_path {fixed_completion_path}"

    # benchmark every path for every model family
    if len(paths) == 1 and len(model_families) == 1 and not force_path_selection:
        logger.info("Overwriting selection strategy because no alternative needs to be evaluated")
        selection_strategy = SelectionStrategy.NONE

    microbenchmark_results = [r for path in paths for r in
                              path.microbenchmark(selection_strategy, validation_removal_attr=validation_removal_attr,
                                                  validation_removal_attr_values=validation_removal_attr_values,
                                                  validation_removal_method=validation_removal_method,
                                                  validation_tuple_removal_table=validation_tuple_removal_table,
                                                  validation_tuple_removal_keep_rate=validation_tuple_removal_keep_rate,
                                                  validation_removal_attr_bias=validation_removal_attr_bias,
                                                  validation_tf_keep_rate=validation_tf_keep_rate,
                                                  max_samples=max_samples)]
    # highest score first
    microbenchmark_results.sort(key=lambda r: r[-1], reverse=True)
    assert microbenchmark_results[0][-1] == max([r[-1] for r in microbenchmark_results])

    mf, path, prefer_baseline, stats, _ = microbenchmark_results[0]
    if prefer_baseline and not force_no_baseline:
        logger.debug("Best path is dominated by baseline. So baseline should be preferred.")
        return None, True

    # if the top path is outgoing, we can safely use this
    if not path.incoming:
        query_plan = path.physical_plan(mf, max_samples=max_samples, fully_synthetic=fully_synthetic,
                                        suppress_nan=suppress_nan, ann_batch_size=ann_batch_size,
                                        ann_neighbors_considered=ann_neighbors_considered,
                                        percentile=percentile, percentile_attributes=percentile_attributes,
                                        predictability_score=predictability_score)
        plan = TopPathUnion([query_plan], [path.rels[-1]], top_path_union_strategy)
    # if it is incoming, we have to find the best plan for every incoming relationship and combine them
    else:
        covered_incoming_rels = set()
        query_plans = []
        final_rels = []
        for mf, path, _, _, _ in microbenchmark_results:
            if path.incoming and path.rels[-1] not in covered_incoming_rels:
                query_plan = path.physical_plan(mf, max_samples=max_samples, fully_synthetic=fully_synthetic,
                                                suppress_nan=suppress_nan, ann_batch_size=ann_batch_size,
                                                ann_neighbors_considered=ann_neighbors_considered,
                                                percentile=percentile, percentile_attributes=percentile_attributes,
                                                predictability_score=predictability_score)
                covered_incoming_rels.add(path.rels[-1])
                query_plans.append(query_plan)
                final_rels.append(path.rels[-1])
        plan = TopPathUnion(query_plans, final_rels, top_path_union_strategy)

    plan.explain()

    return plan, False, stats


class CompletionPath:
    def __init__(self, schema, start_table, rels, model_families, requested_tables, target_relationships):
        self.schema = schema
        self.start_table = start_table
        self.rels = rels
        self.model_families = model_families
        self.requested_tables = requested_tables
        self.target_relationships = target_relationships

        self.tables = set()
        if target_relationships is not None:
            self.tables.update([r.incoming_table for r in target_relationships])
            self.tables.update([r.outgoing_table for r in target_relationships])
            for r in target_relationships:
                if r not in self.rels:
                    self.rels.append(r)
        self.tables.update([r.incoming_table for r in self.rels])
        self.tables.update([r.outgoing_table for r in self.rels])

        self.attribute_names = set()
        for t in self.tables:
            self.attribute_names.update([a.full_name for a in t.attributes])

    def microbenchmark(self, selection_strategy, validation_removal_attr=None, validation_removal_attr_values=None,
                       validation_removal_method=None, validation_tuple_removal_table=None,
                       validation_tuple_removal_keep_rate=None, validation_removal_attr_bias=None,
                       validation_tf_keep_rate=None, max_samples=None):

        prefer_baseline = False

        if selection_strategy == SelectionStrategy.ARTIFICIAL_BIAS or \
                selection_strategy == SelectionStrategy.ARTIFICIAL_BIAS_MAE or \
                selection_strategy == SelectionStrategy.UNIFORM_MAE:

            scores, stats = self.compute_scores(selection_strategy, validation_removal_attr,
                                                validation_removal_attr_bias,
                                                validation_removal_attr_values, validation_removal_method,
                                                validation_tf_keep_rate, validation_tuple_removal_keep_rate,
                                                validation_tuple_removal_table, max_samples)
            if np.all(scores < 0):
                prefer_baseline = True

        elif selection_strategy == SelectionStrategy.COMBINED:

            mae_scores, stats = self.compute_scores(SelectionStrategy.UNIFORM_MAE, validation_removal_attr,
                                                    validation_removal_attr_bias,
                                                    validation_removal_attr_values, validation_removal_method,
                                                    validation_tf_keep_rate, validation_tuple_removal_keep_rate,
                                                    validation_tuple_removal_table, max_samples)

            mae_scores += np.min(mae_scores)
            mae_scores /= mae_scores
            assert np.all(0 <= mae_scores) and np.all(mae_scores <= 1)

            ab_scores, stats = self.compute_scores(SelectionStrategy.ARTIFICIAL_BIAS, validation_removal_attr,
                                                   validation_removal_attr_bias,
                                                   validation_removal_attr_values, validation_removal_method,
                                                   validation_tf_keep_rate, validation_tuple_removal_keep_rate,
                                                   validation_tuple_removal_table, max_samples)
            if np.all(ab_scores < 0):
                prefer_baseline = True

            scores = mae_scores + ab_scores

        elif selection_strategy == SelectionStrategy.NONE:
            scores = [0 for mf in self.model_families]
            stats = [None for mf in self.model_families]
        else:
            raise NotImplementedError

        benchmark_results = []
        for mf, score, stat in zip(self.model_families, scores, stats):
            benchmark_results.append((mf, self, prefer_baseline, stat, score))

        return benchmark_results

    def compute_scores(self, selection_strategy, validation_removal_attr, validation_removal_attr_bias,
                       validation_removal_attr_values, validation_removal_method, validation_tf_keep_rate,
                       validation_tuple_removal_keep_rate, validation_tuple_removal_table, max_samples):
        scores = []
        stats = []
        if selection_strategy == SelectionStrategy.ARTIFICIAL_BIAS_MAE or \
                selection_strategy == SelectionStrategy.ARTIFICIAL_BIAS:
            assert validation_removal_attr is not None
            a_mapping, r_mapping, t_mapping, validation_schema = \
                self.more_incomplete_schema(validation_removal_attr, validation_removal_attr_bias,
                                            validation_removal_attr_values, validation_removal_method,
                                            validation_tf_keep_rate, validation_tuple_removal_keep_rate,
                                            validation_tuple_removal_table)

        else:
            a_mapping, r_mapping, t_mapping, validation_schema = \
                self.more_incomplete_schema(validation_removal_attr, validation_removal_attr_bias,
                                            validation_removal_attr_values,
                                            [RemovalMethod.UNIFORM] * len(validation_tuple_removal_table),
                                            validation_tf_keep_rate, validation_tuple_removal_keep_rate,
                                            validation_tuple_removal_table)
        for mf in self.model_families:
            restrict_requested = True
            if selection_strategy == SelectionStrategy.UNIFORM_MAE or \
                    selection_strategy == SelectionStrategy.ARTIFICIAL_BIAS_MAE:
                restrict_requested = False
            S, mapped_query = self.completion_path_plan(a_mapping, mf, r_mapping, t_mapping, validation_schema,
                                                        restrict_requested=restrict_requested, max_samples=max_samples)

            # find index that is now relevant
            rel_idx = [i for i, a in enumerate(validation_removal_attr) if a in self.attribute_names][0]

            if selection_strategy == SelectionStrategy.ARTIFICIAL_BIAS:
                pred_baseline, pred_mean, actual_mean, actual_no_tuples, baseline_no_tuples, sum_tuples = \
                    evaluate_rerr_reduction(mapped_query, S.df_rows, S.weights, validation_removal_method[rel_idx],
                                            validation_removal_attr[rel_idx], [validation_removal_attr_values[rel_idx]],
                                            restrict_connected=True)
                score = rel_err_reduction(pred_baseline, pred_mean, actual_mean)
                stats.append([pred_baseline, pred_mean, actual_mean, score])
            else:
                # higher is better, so negative absolute error
                score = -mae(mapped_query, S.df_rows, S.weights, self.start_table.primary_key[0].full_name,
                             validation_removal_attr[rel_idx], validation_removal_method[rel_idx],
                             [validation_removal_attr_values[rel_idx]],
                             restrict_connected=False)
                stats.append(score)
            scores.append(score)

        return np.array(scores), stats

    def completion_path_plan(self, a_mapping, mf, r_mapping, t_mapping, validation_schema, restrict_requested=True,
                             max_samples=None, top_path_union_strategy=UnionStrategy.COMBINE):
        mapped_target_r = None
        if self.target_relationships is not None:
            mapped_target_r = [r_mapping[r] for r in self.target_relationships]
        mf_mapped = mf.map_to_validation_schema(t_mapping, r_mapping, a_mapping)

        tables = self.requested_tables
        if not restrict_requested:
            tables = self.tables
        query_plan = self._physical_plan(mf_mapped, requested_tables=[t_mapping[t] for t in tables],
                                         start_table=t_mapping[self.start_table],
                                         rels=[r_mapping[r] for r in self.rels], target_relationships=mapped_target_r,
                                         max_samples=max_samples, fully_synthetic=True)
        plan = TopPathUnion([query_plan], [r_mapping[self.rels[-1]]], top_path_union_strategy=top_path_union_strategy)
        S, _ = plan.execute()
        mapped_query = Query(validation_schema, [t.name for t in tables])
        return S, mapped_query

    def more_incomplete_schema(self, validation_removal_attr, validation_removal_attr_bias,
                               validation_removal_attr_values, validation_removal_method, validation_tf_keep_rate,
                               validation_tuple_removal_keep_rate, validation_tuple_removal_table):
        # the new incomplete schema is derived from the previously incomplete schema and even more incomplete
        validation_tables = [t.name for t in self.schema.tables]
        validation_schema, t_mapping, r_mapping, a_mapping = _incomplete_schema(None, validation_tables,
                                                                                self.schema,
                                                                                return_mapping=True)
        # the previously incomplete tables are now considered complete
        for t in validation_schema.tables:
            original_table = {v: k for k, v in t_mapping.items()}[t]

            orig_complete_dataset = original_table.complete_dataset
            orig_remaining_pks = original_table.remaining_pks

            df_table = orig_complete_dataset.df_rows
            pk_name = t.primary_key[0].full_name
            df_table = df_table[df_table[pk_name].isin(orig_remaining_pks)]
            t.complete_dataset = Dataset(df_table, [a_mapping[a] for a in orig_complete_dataset.attributes])
            t.complete_pks = copy(orig_remaining_pks)
            t.remaining_pks = orig_remaining_pks
            t._incomplete_dataset = t.complete_dataset
        derive_incomplete_schema(cascading_deletes=True, removal_attrs=validation_removal_attr,
                                 removal_attr_biases=validation_removal_attr_bias,
                                 removal_attr_values=validation_removal_attr_values,
                                 removal_methods=validation_removal_method, incomplete_schema=validation_schema,
                                 tf_keep_rates=[validation_tf_keep_rate for r in self.rels],
                                 tf_removals=[r.tf_attribute.full_name for r in self.rels],
                                 tuple_removal_keep_rates=validation_tuple_removal_keep_rate,
                                 tuple_removal_tables=validation_tuple_removal_table)
        return a_mapping, r_mapping, t_mapping, validation_schema

    def physical_plan(self, mf, max_samples=None, fully_synthetic=True, suppress_nan=False, ann_batch_size=10000,
                      ann_neighbors_considered=100000, percentile=None, percentile_attributes=None,
                      predictability_score=None):
        return self._physical_plan(mf, self.requested_tables, self.start_table, self.rels, self.target_relationships,
                                   max_samples=max_samples, fully_synthetic=fully_synthetic, suppress_nan=suppress_nan,
                                   ann_batch_size=ann_batch_size, ann_neighbors_considered=ann_neighbors_considered,
                                   percentile=percentile, percentile_attributes=percentile_attributes,
                                   predictability_score=predictability_score)

    @staticmethod
    def _physical_plan(mf, requested_tables, start_table, rels, target_relationships,
                       max_samples=None, fully_synthetic=True, suppress_nan=False, ann_batch_size=10000,
                       ann_neighbors_considered=100000, percentile=None, percentile_attributes=None,
                       predictability_score=None):
        all_rels = rels
        if target_relationships is not None:
            all_rels = set(all_rels)
            all_rels.update(target_relationships)
            all_rels = list(all_rels)

        assert len(all_rels) > 0, "The length must be >= 0"

        joined_tables, all_rels = relationship_order(all_rels, return_tables=True, first_table=start_table)

        query_plan = [LoadCompleteTable(start_table)]
        ann_buffer = []
        final_projection = ProjectRequestedJoin(all_rels, requested_tables)
        # todo if we have a percentile: add tuple factors as attribute if we are interested in COUNT/SUM queries. For
        # normalizing tuples set inverse = True (which will minimize them for the upper bound of the conf. interval)

        for i, r in enumerate(all_rels):
            inverse = joined_tables[i] == r.outgoing_table
            if (r, inverse) not in mf.r_model_dict.keys():
                raise ValueError(f'No model learned for {str(r)}, inverse={inverse}, '
                                 f'requested_tables {[str(t) for t in requested_tables]}, start_table {start_table}')

            model = mf.r_model_dict[(r, inverse)]
            query_plan.append(JoinRelationship(r, inverse=inverse, model=model, max_samples=max_samples,
                                               suppress_nan=suppress_nan, percentile=percentile,
                                               percentile_attributes=percentile_attributes,
                                               predictability_score=predictability_score))
            if not fully_synthetic:
                # no fan out, add rel to all anns required
                if inverse:
                    # add this relationship as evidence
                    for ann_r, ann_inverse, ann_replace_join_relationships in ann_buffer:
                        ann_replace_join_relationships.append(r)
                # now fan out cannot
                else:
                    for ann_r, ann_inverse, ann_replace_join_relationships in ann_buffer:
                        query_plan.append(ANN_Replacement(ann_r, ann_inverse, ann_replace_join_relationships,
                                                          batch_size=ann_batch_size,
                                                          neighbors_considered=ann_neighbors_considered,
                                                          fan_out_tuple_factors=final_projection.fan_out_tuple_factors))
                ann_buffer.append((r, inverse, []))

        if not fully_synthetic:
            for ann_r, ann_inverse, ann_replace_join_relationships in ann_buffer:
                query_plan.append(
                    ANN_Replacement(ann_r, ann_inverse, ann_replace_join_relationships, batch_size=ann_batch_size,
                                    neighbors_considered=ann_neighbors_considered,
                                    fan_out_tuple_factors=final_projection.fan_out_tuple_factors))

        query_plan.append(final_projection)
        return query_plan

    @property
    def incoming(self):
        return self.rels[-1].incoming_table in self.requested_tables


def find_requested_tables(query):
    requested_tables = set()
    if query.target_relationships is not None:
        assert query.single_target_table is None
        for r in query.target_relationships:
            requested_tables.add(r.outgoing_table)
            requested_tables.add(r.incoming_table)
    else:
        requested_tables = {query.single_target_table}
    return requested_tables
