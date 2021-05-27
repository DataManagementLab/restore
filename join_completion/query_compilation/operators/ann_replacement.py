import logging
from time import perf_counter

import numpy as np
import scipy
from numba import njit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from join_completion.query_compilation.operators.ann_index import NMSLibIndex
from join_completion.query_compilation.operators.operator import Incomplete_Join_Operation
from schema_setup.data_preparation.utils import batch
from schema_setup.schema.dataset import Dataset
from schema_setup.schema.schema import Attribute
from schema_setup.schema.schema_utils import extend_by_rels

logger = logging.getLogger(__name__)


@njit
def greedy_assign(cdist, repeat_idx, desired_batch_size):
    assignment_mask = np.ones(cdist.shape[1], dtype=np.bool_)
    assignments = np.empty(len(repeat_idx), dtype=np.int64)

    no_batches = np.ceil(repeat_idx.shape[0] / desired_batch_size)
    batch_size = np.int(repeat_idx.shape[0] / no_batches)
    y_batch_size = np.int(cdist.shape[1] / no_batches)

    for i, r_idx in enumerate(repeat_idx):
        min_dist = np.inf
        batch_no = np.int(np.floor(i / batch_size))

        start_idx = batch_no * y_batch_size
        end_idx = min(start_idx + y_batch_size, cdist.shape[1])
        # for j in range(cdist.shape[1]):
        for j in range(start_idx, end_idx):
            if not assignment_mask[j]:
                continue
            if cdist[r_idx, j] < min_dist:
                assignments[i] = j
                min_dist = cdist[r_idx, j]
        # fallback for edge cases
        if np.isinf(min_dist):
            for j in range(cdist.shape[1]):
                if not assignment_mask[j]:
                    continue
                if cdist[r_idx, j] < min_dist:
                    assignments[i] = j
                    min_dist = cdist[r_idx, j]
        assignment_mask[assignments[i]] = False

    return assignments


class ANN_Replacement(Incomplete_Join_Operation):
    def __init__(self, r, inverse, replace_join_relationships, batch_size=1000, neighbors_considered=10000,
                 fan_out_tuple_factors=None):
        Incomplete_Join_Operation.__init__(self)
        self.r = r
        self.inverse = inverse
        self.replace_join_relationships = replace_join_relationships
        self.batch_size = batch_size
        self.neighbors_considered = neighbors_considered
        self.stats.update({
            'replaced_tuples': 0,
            'input_tuples': 0,
            'elapsed_time': 0,
            'step': str(self)
        })
        self.fan_out_tuple_factors = fan_out_tuple_factors
        if self.fan_out_tuple_factors is None:
            self.fan_out_tuple_factors = []

    def execute(self, current_join):
        if self.inverse:
            return self.inverse_ann(current_join)
        else:
            return self.ann(current_join)

    def ann(self, current_join):
        next_table = self.r.outgoing_table

        # Tuples without foreign key, join with current_join
        single_right_tuples = next_table.incomplete_dataset.df_rows[
            next_table.incomplete_dataset.df_rows[self.r.outgoing_attributes[0].full_name].isna()]
        assert len(self.r.pks_without_fk) == len(single_right_tuples)
        if len(single_right_tuples) > 0:
            assert not next_table.complete, "This is inconsistent. Empty foreign keys mean that the referenced tuple " \
                                            "is not known."
            # we could use LSH to replace right tuples in synthesized_df
            # raise NotImplementedError("Some technique like LocalitySensitiveHashing is required here")

        return current_join

    def inverse_ann(self, current_join, verbose=False):
        """
        Replaces synthesized tuples with known tuples for n:1 join (e.g., Orders->Customers)

        Assumption:
            - Synthesized orders: could be existing customer or synthesized customer
            - Non-synthesized orders: could be existing customer or synthesized customer
            - current multiplication factor is unbiased estimator for actual mf (defined below)

        Steps:
            - Load customers and compute current tf
            - we have to estimate for every customer how often it appears in the join
                - appearance = multiplication_factor * tf
                - def multiplication_factor: how often does cust_id appear in join/ tf
                - multiplication_factor can be unknown in case we do not have all corresponding orders, we approximate
                    by current_tf/current_appearance
                - for some customer we obtain exp_appearance
                - learn a ML model predicting exp_appearance for other customers
            - Look at customers with (tf < current_tf or tf is nan)
                - compute missing_appearance as current_appearance - exp_appearance
                - left join of customers with additional evidence
                - Learn Approximate Nearest Neighbor Index over Synthesized tuples
                - Replace Nearest Neighbor with customer (as often as missing_appearance, do not use synthesized tuple
                    twice)
                - if complete and still synthesized tuples: continue

        :param current_join:
        :return:
        """
        inverse_ann_start_t = perf_counter()
        input_tuples = len(current_join.df_rows)

        # Compute current tf
        next_table = self.r.incoming_table

        next_table_dataset = Dataset(next_table.incomplete_dataset.df_rows, next_table.incomplete_dataset.attributes)
        next_table_dataset = next_table_dataset.augment_current_tuple_factors(self.r.outgoing_table.incomplete_dataset,
                                                                              self.r)

        # create backup of tuple factors (in case we replaced with NaN)
        tf_backup = dict()
        for a in current_join.attributes:
            if a.is_tf:
                tf_backup[a] = np.copy(current_join.df_rows[a.full_name].values)

        # compute how often next table currently appears in join
        next_pk = next_table.primary_key[0].full_name
        curr_appearance = current_join.df_rows.groupby([next_pk]).count()
        curr_appearance = curr_appearance.iloc[:, 0]
        curr_appearance = curr_appearance.reset_index(drop=False)
        curr_appearance.columns = [next_pk, 'current_appearance']

        # compute how often we expect to see this tuple in the join
        next_df_rows = next_table_dataset.df_rows
        next_df_rows = next_df_rows.merge(curr_appearance, left_on=next_pk, right_on=next_pk, how='left')
        next_df_rows['current_appearance'].fillna(0, inplace=True)
        next_df_rows['multiplication_factor'] = next_df_rows['current_appearance'] / next_df_rows['current_tf']
        next_df_rows.loc[next_df_rows['current_tf'] == 0, 'multiplication_factor'] = 1
        next_df_rows['multiplication_factor'].fillna(1, inplace=True)
        next_df_rows['expected_appearance'] = next_df_rows[self.r.tf_attribute.full_name] * next_df_rows[
            'multiplication_factor']
        next_df_rows.loc[next_df_rows[self.r.tf_attribute.full_name] == 0, 'expected_appearance'] = 0

        # add expected appearance to attributes
        next_table_dataset.attributes.remove(next_table_dataset.attribute_dict['current_tf'])
        next_df_rows.drop(columns=['multiplication_factor', 'current_tf'], inplace=True)
        next_table_dataset.df_rows = next_df_rows
        next_table_dataset.attributes.append(Attribute(None, 'current_appearance'))
        next_table_dataset.attributes.append(Attribute(None, 'expected_appearance'))
        assert len(next_table_dataset.attributes) == len(next_table_dataset.df_rows.columns)

        if np.all(~np.isnan(next_table_dataset.df_rows['expected_appearance'])) and \
                np.allclose(next_table_dataset.df_rows['expected_appearance'] -
                            next_table_dataset.df_rows['current_appearance'], 0):
            # nothing to do
            self.stats.update({
                'replaced_tuples': 0,
                'input_tuples': input_tuples,
                'elapsed_time': perf_counter() - inverse_ann_start_t,
            })
            return current_join

        # movie.tf_movie_actor.movie_id
        next_table_dataset = extend_by_rels(next_table_dataset, self.replace_join_relationships, how='inner')
        # if tf is zero for any join we also do this tuple should not be replaced
        for tf_attribute in self.fan_out_tuple_factors:
            if tf_attribute.full_name in next_table_dataset.df_rows.columns:
                next_table_dataset.df_rows.loc[next_table_dataset.df_rows[tf_attribute.full_name] == 0,
                                               'expected_appearance'] = 0

        if not np.all(~np.isnan(next_table_dataset.df_rows['expected_appearance'])):
            proj_scopes = [i for i, a in enumerate(next_table_dataset.attributes) if not (a.is_pk or a.is_fk)]
            training_data = next_table_dataset.df_rows.iloc[:, proj_scopes]

            replace_idxs = next_table_dataset.df_rows['expected_appearance'].isna() | np.isinf(
                next_table_dataset.df_rows['expected_appearance'])
            test_data = training_data[replace_idxs]
            training_data = training_data[~replace_idxs]
            Y = training_data['expected_appearance']
            proj_scopes = [i for i, col in enumerate(training_data.columns) if str(col) != 'expected_appearance']
            X = training_data.iloc[:, proj_scopes]

            start_t = perf_counter()
            exp_app_model = xgb.XGBRegressor().fit(X, Y)
            logger.info(f"Learned xgb model for {len(X)} tuples in {perf_counter() - start_t:.2f} secs")

            prediction = exp_app_model.predict(test_data.iloc[:, proj_scopes])
            prediction = np.clip(prediction, next_table_dataset.df_rows.loc[replace_idxs, 'current_appearance'], np.inf)
            prediction = prediction.astype(int)
            next_table_dataset.df_rows.loc[replace_idxs, 'expected_appearance'] = prediction

        # how often does the tuple get imputed?
        no_imputations = next_table_dataset.df_rows['expected_appearance'] - next_table_dataset.df_rows[
            'current_appearance']
        # make sure we are really dealing with integers
        no_imputations = no_imputations.values.astype(int)
        no_imputations = np.clip(no_imputations, 0, np.inf)
        original_no_imputations = np.copy(no_imputations)

        # project to relevant tuples
        next_table_dataset = next_table_dataset.project(
            [a for a in current_join.attributes if a in next_table_dataset.attributes])
        for a in next_table_dataset.attributes:
            if a.is_pk:
                if np.any(np.isnan(next_table_dataset.df_rows[a.full_name])):
                    raise NotImplementedError("Nan-handling not correctly implemented for imputation - is not hard "
                                              "however. We would just do several passes removing nan-parts of the "
                                              "imputation attributes")
        imp_rows = next_table_dataset.df_rows.values
        assert len(imp_rows) == len(no_imputations)
        imp_ann_projection = [i for i, a in enumerate(next_table_dataset.attributes) if not (a.is_pk or a.is_fk)]
        ann_scopes = [i for i, a in enumerate(current_join.attributes) if a in next_table_dataset.attributes and
                      not (a.is_pk or a.is_fk)]
        imp_scopes = [i for i, a in enumerate(current_join.attributes) if a in next_table_dataset.attributes]

        start_t = perf_counter()

        total_no_imp = np.sum(no_imputations)
        t_nn = None
        t_ann = None
        total_processed = 0
        assert np.all(no_imputations >= 0)
        # call once with dummy arguments to exclude compilation time from first measurement
        greedy_assign(np.array([[1., 1.]]), np.array([0, 0]), 2)

        while True:

            remaining_imp = np.where(no_imputations > 0)[0]
            np.random.shuffle(remaining_imp)
            if not len(remaining_imp) > 0:
                if next_table.complete:
                    logger.info(f"Reset no of imputations since table is complete")
                    no_imputations = original_no_imputations
                    remaining_imp = np.where(no_imputations > 0)[0]
                    np.random.shuffle(remaining_imp)
                    total_processed += np.sum(original_no_imputations)
                else:
                    break
            logger.info(f"Replacing {len(remaining_imp)} synthetic tuples using ANN")

            synthetic_row_ids = np.where(current_join.df_rows[next_pk].values < 0)[0]
            if len(synthetic_row_ids) == 0:
                logger.warning(
                    "Had to stop the replacement since there are no more synthetic rows that can be replaced")
                break

            # benchmark how long pairwise step would take
            # or the last ann step took longer than this benchmark (performance deteriorates over time)
            if t_nn is None or (t_ann is not None and t_ann > t_nn):
                # some tuples need several imputations, we repeat them as often as required. To still have the correct
                # batch size we compute the cumulative sum here
                imps = np.cumsum(no_imputations[remaining_imp])
                batch = remaining_imp
                if len(imps) == 0:
                    logger.info(f"Finished imputation")
                    break

                if imps[-1] > self.batch_size:
                    curr_batch_idx = np.where(imps > self.batch_size)[0][0] + 1
                    batch = remaining_imp[:curr_batch_idx]

                t_nn = self.pairwise_nn_iteration(ann_scopes, current_join, imp_ann_projection,
                                                  batch, imp_rows, imp_scopes, no_imputations,
                                                  synthetic_row_ids, neighbors_considered=self.neighbors_considered)
                logger.info(f"t_nn {t_nn * 1000:.2f}ms")
            else:
                t_ann = self.approximate_nn_iteration(ann_scopes, current_join, imp_ann_projection, imp_rows,
                                                      imp_scopes, next_pk, no_imputations, remaining_imp,
                                                      synthetic_row_ids, verbose,
                                                      neighbors_considered=self.neighbors_considered,
                                                      batch_size=self.batch_size)
                logger.info(f"t_ann {t_ann * 1000:.2f}ms")

            total_time = perf_counter() - start_t
            sum_processed = total_no_imp - np.sum(no_imputations)
            logger.info(f"Replacement took {total_time} secs. "
                        f"\n\t{sum_processed:.0f}/{total_no_imp:.0f} "
                        f"({sum_processed / total_no_imp * 100:.2f}%) processed."
                        f"\n\tAvg time per imputation: {total_time / sum_processed * 1000:.2f}ms")
            assert np.all(no_imputations >= 0)

        # replace nan tfs with backup
        for a in current_join.attributes:
            if a.is_tf:
                null_idxs = current_join.df_rows[a.full_name].isna()
                backup_rows = tf_backup[a][null_idxs]
                current_join.df_rows.loc[null_idxs, a.full_name] = backup_rows

        assert len(current_join.attributes) == len(current_join.df_rows.columns)
        self.stats.update({
            'replaced_tuples': total_processed + (total_no_imp - np.sum(no_imputations)),
            'input_tuples': input_tuples,
            'elapsed_time': perf_counter() - inverse_ann_start_t,
        })

        return current_join

    def pairwise_nn_iteration(self, ann_scopes, current_join, imp_ann_projection, imp_batch, imp_rows, imp_scopes,
                              no_imputations, synthetic_row_ids, neighbors_considered):
        start_t = perf_counter()
        np.random.shuffle(synthetic_row_ids)
        curr_synthetic_row_ids = synthetic_row_ids[
                                 :int(neighbors_considered / self.batch_size * np.sum(no_imputations[imp_batch]))]
        syn_data = current_join.df_rows.values[curr_synthetic_row_ids][:, ann_scopes]
        imp_data = imp_rows[imp_batch][:, imp_ann_projection]
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(syn_data)
        syn_data = imputer.transform(syn_data)
        imp_data = imputer.transform(imp_data)
        scaler = StandardScaler().fit(syn_data)
        syn_data = scaler.transform(syn_data)
        imp_data = scaler.transform(imp_data)
        # we cannot impute more than number of synthetic tuples
        imp_data = imp_data[:len(syn_data)]
        imp_batch = imp_batch[:len(syn_data)]
        # compute pairwise distances
        cdist = scipy.spatial.distance.cdist(imp_data, syn_data, metric='euclidean')

        # repeat for tuples that appear more than once
        repeats = no_imputations[imp_batch].astype(int)
        repeat_idx = np.repeat(np.arange(cdist.shape[0]), repeats=repeats)
        repeat_idx = repeat_idx[:cdist.shape[1]]

        # this is not very memory efficient
        # cdist = np.repeat(cdist, repeats=repeats, axis=0)
        # imp_rows = np.repeat(imp_rows[imp_batch], repeats=repeats, axis=0)
        # imp_batch = np.repeat(imp_batch, repeats=repeats, axis=0)

        logger.info(f"Distance computation took {perf_counter() - start_t} secs. ")
        logger.info(f"PNN: Average distance: {np.average(cdist)}")
        ass_start_t = perf_counter()
        assert not np.any(np.isnan(cdist))
        assert np.all(np.isfinite(cdist))
        # can this item still be assigned?
        assignments = greedy_assign(cdist, repeat_idx, self.batch_size)
        logger.info(f"Assignment of {len(assignments)} tuples ({cdist.shape[1]} syn tuples) "
                    f"took {perf_counter() - ass_start_t} secs. ")

        current_join.df_rows.iloc[curr_synthetic_row_ids[assignments], imp_scopes] = imp_rows[imp_batch[repeat_idx]]
        replaced_idx, counts = np.unique(imp_batch[repeat_idx], return_counts=True)
        no_imputations[replaced_idx] -= counts

        total_time = perf_counter() - start_t
        return total_time / len(repeat_idx)

    def approximate_nn_iteration(self, ann_scopes, current_join, imp_ann_projection, imp_rows, imp_scopes, next_pk,
                                 no_imputations, remaining_imp, synthetic_row_ids, verbose, neighbors_considered=100000,
                                 batch_size=10000):
        start_t = perf_counter()
        # index = FaissIndex(gpu=True, sampling_threshold=100000)#, list=int(10 * np.sqrt(len(synthetic_row_ids))))
        index = NMSLibIndex(sampling_threshold=neighbors_considered, verbose=verbose)
        index.fit(synthetic_row_ids, current_join.df_rows.values[synthetic_row_ids][:, ann_scopes])

        replaced_tuples = 1
        for batch_no, imp_batch in enumerate(batch(remaining_imp, batch_size=batch_size)):

            # indexes in synthetic_row_ids that should be replaced by imp rows
            replace_idx, distances = index.query_batch(imp_rows[imp_batch][:, imp_ann_projection], n=20)
            # only consider non-duplicate rows
            _, non_red_idx = np.unique(replace_idx, return_index=True)
            duplicate_predictions = (1 - len(non_red_idx) / len(replace_idx))

            # out of these only consider synthetic rows which have not already been replaced
            rel_idx = [i for i in non_red_idx if current_join.df_rows[next_pk].iloc[replace_idx[i]] < 0]
            duplicate_batch_replaces = (1 - len(rel_idx) / len(non_red_idx))

            if duplicate_batch_replaces > 0.9:
                break

            if verbose:
                logger.info(f"Duplicate predictions reduced by {duplicate_predictions * 100:.2f}%")
                logger.info(f"Already replaced reduced by {duplicate_batch_replaces * 100:.2f}%")
                if batch_no % 10 == 0:
                    logger.info(f"{batch_no} batches took {perf_counter() - start_t} secs.")

            # finally do the legal imputation
            replaced_tuples += len(replace_idx[rel_idx])
            current_join.df_rows.iloc[replace_idx[rel_idx], imp_scopes] = imp_rows[imp_batch[rel_idx]]
            logger.info(f"ANN: Average assigned distance {np.average(distances[rel_idx]):.2f}")

            # denote that these tuples were imputed
            no_imputations[imp_batch[rel_idx]] -= 1

        total_time = perf_counter() - start_t
        return total_time / replaced_tuples

    def step_name(self):
        return f'ANN{self.batch_size}{self.neighbors_considered}{str(self.r)}' \
               f'{"_".join([str(r) for r in self.replace_join_relationships])}'

    def __str__(self):
        return f'ApproximateNearestNeighbor(batch_size={self.batch_size}, ' \
               f'neighbors_considered={self.neighbors_considered}, ' \
               f'r={str(self.r)}, ' \
               f'replace_join_relationships={",".join([str(r) for r in self.replace_join_relationships])})'
