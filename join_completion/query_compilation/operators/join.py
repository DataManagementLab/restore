import logging
from time import perf_counter

import numpy as np

from join_completion.query_compilation.operators.operator import Incomplete_Join_Operation

logger = logging.getLogger(__name__)


class JoinRelationship(Incomplete_Join_Operation):
    def __init__(self, r, inverse, model=None, max_samples=None, suppress_nan=False, percentile=None,
                 percentile_attributes=None, predictability_score=None):
        Incomplete_Join_Operation.__init__(self)
        self.r = r
        self.inverse = inverse
        self.model = model
        self.suppress_nan = suppress_nan
        self.percentile = percentile
        self.predictability_score = predictability_score
        self.percentile_attributes = percentile_attributes

        self.max_samples = max_samples
        if self.max_samples is not None:
            logger.warning("Sampling is solely intended for debugging. It will produce biased estimates when used for "
                           "experiments")
        self.stats.update({
            'output_tuples': 0,
            'input_tuples': 0,
            'elapsed_time': 0,
            'step': str(self),
        })

    def __str__(self):
        model_desc = 'None'
        if self.model is not None:
            model_desc = self.model.model_name

        percentile_suffix = ''
        if self.percentile is not None:
            percentile_suffix = f', percentile={self.percentile} ({self.predictability_score})'

        return f'JoinRelationship({str(self.r)}, inverse={self.inverse}, model={model_desc}{percentile_suffix})'

    def execute(self, current_join):
        start_t = perf_counter()
        input_tuples = len(current_join.df_rows)

        if self.inverse:
            S = self.inverse_join(current_join)
        else:
            S = self.join(current_join)

        self.stats.update({
            'output_tuples': len(S.df_rows),
            'input_tuples': input_tuples,
            'elapsed_time': perf_counter() - start_t,
        })

        return S

    def join(self, current_join):
        """
        Executes 1:n join (e.g., Customers->Orders)
        Situation:
            - For some customers we know complete orders -> join directly
            - For some we only know partial or no orders

        Steps:
            (1) Join orders for which we have customer_id
            (2) Synthesize orders for incomplete customer sets
            (3a) If naive: done
            (3b) Not naive:
                - for some orders we know how often it was used in the join (predict for remaining)
                - for every order wo cust_id
                    - predict how often joined
                    - find NN using LSH, replace
                - old: (- for order predict likely cust_id, does not work if cust_id not seen during training)
                - complete: continue to join existing orders

        :param current_join:
        :return:
        """
        # (1) Join remaining tuples
        next_table = self.r.outgoing_table.incomplete_dataset

        remaining_join = current_join.join(next_table, self.r, how='inner')
        # (1b) sample tuple factors
        self.predict_nan_tfs(remaining_join, self.r.outgoing_table, ge_zero=True)

        # (2) Synthesize orders for incomplete customer sets
        current_join = current_join.augment_current_tuple_factors(next_table, self.r)
        # take current tuple factors into account
        synthesized = self.model.complete_1_n(current_join, next_table, self.r, max_samples=self.max_samples,
                                              suppress_nan=self.suppress_nan, virtual_ids=True, keep_ids=True,
                                              percentile=self.percentile,
                                              percentile_attributes=self.percentile_attributes,
                                              predictability_score=self.predictability_score
                                              )

        # append complete join and synthesized join
        remaining_join = remaining_join.concat(synthesized)
        self.check_valid(remaining_join)

        return remaining_join

    def check_valid(self, current_join):
        assert len(current_join.weights) == len(current_join.df_rows)

        for i, a in enumerate(current_join.attributes):
            # do not check every tuple factor since we only care about joins we actually execute (below)
            if a.is_pk:
                assert not np.any(np.isnan(current_join.df_rows.values[:, i]))
            if a == self.r.tf_attribute:
                # tuple factors in a join (which is not left outer should all be >= 0
                assert np.all(current_join.df_rows.values[:, i] >= 1), f"Tuple factor {a.full_name} has zero-values"

    def restrict_to_incomplete_sets(self, current_join):
        # filter out tuples where we know that they have all partners in the join
        rel_rows = ~current_join.df_rows[self.r.incoming_attributes[0].full_name].isin(self.r.complete_set_pks)
        current_join.df_rows = current_join.df_rows[rel_rows]
        current_join.weights = current_join.weights[rel_rows]

        return current_join

    def inverse_join(self, current_join):
        """
        Executes n:1 join (e.g., Orders->Customers)

        Situation:
            - For any order we do not know corresponding customer (cust_id not set)
            - For some customers we have the corresponding orders
            - For the complete customers-orders we know how often the customers appear in the join

        Steps:
            (1a) Join orders for which we have customer_id
            (1b) Synthesize tuple factors that are nan in this join
            (2) Synthesize customers for remaining orders
            (3a) If naive: done
            (3b) Not naive:
                - Predict how often customer appears in Join
                - LSH to replace synthetic customer with this
                - If customers is complete: repeat until all replaced

        :param current_join:
        :return:
        """
        assert len(current_join.weights) == len(current_join.df_rows)

        # (1a) Join remaining tuples
        next_table = self.r.incoming_table.incomplete_dataset
        remaining_join = current_join.join(next_table, self.r, how='inner')

        # (1b) sample tuple factors
        self.predict_nan_tfs(remaining_join, self.r.incoming_table, ge_zero=True)

        # (2) Synthesize customers for orders without customer
        rel_rows = current_join.df_rows[self.r.outgoing_attributes[0].full_name].isna()
        current_join.df_rows = current_join.df_rows[rel_rows]
        current_join.weights = current_join.weights[rel_rows]

        synthesized = self.model.complete_n_1(current_join, next_table, self.r, max_samples=self.max_samples,
                                              suppress_nan=self.suppress_nan, virtual_ids=True, keep_ids=True,
                                              percentile=self.percentile,
                                              percentile_attributes=self.percentile_attributes,
                                              predictability_score=self.predictability_score)

        # append complete join and synthesized join
        current_join = remaining_join.concat(synthesized)
        self.check_valid(current_join)

        return current_join

    def predict_nan_tfs(self, remaining_join, completion_table, ge_zero=True):
        replace_idxs = remaining_join.df_rows[self.r.tf_attribute.full_name].isna()
        tfs_pred = self.model.predict_tf(self.r, completion_table, remaining_join, predict_idx=replace_idxs,
                                         ge_zero=ge_zero, percentile=self.percentile,
                                         percentile_attributes=self.percentile_attributes,
                                         predictability_score=self.predictability_score)
        logger.info(f"Predicting {len(tfs_pred)} tuple factors")
        remaining_join.df_rows.loc[replace_idxs, self.r.tf_attribute.full_name] = tfs_pred

    def step_name(self):
        model_desc = 'None'
        if self.model is not None:
            model_desc = self.model.model_name
        return f'J{str(self.r)}{model_desc}{str(self.max_samples)}{self.suppress_nan}'
