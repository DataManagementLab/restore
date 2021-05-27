import logging
from time import perf_counter

import numpy as np

from join_completion.query_compilation.operators.operator import Incomplete_Join_Operation
from schema_setup.schema.schema_utils import custom_bfs

logger = logging.getLogger(__name__)


class ProjectRequestedJoin(Incomplete_Join_Operation):
    def __init__(self, joined_relationships, requested_tables):
        Incomplete_Join_Operation.__init__(self)
        self.joined_relationships = joined_relationships
        self.requested_tables = requested_tables

        # find fan-out tuple factors
        self.fan_out_tuple_factors = []

        def tuple_factor_fan_out(table=None, visited_tables=None, fan_out_tuple_factors=None, req_tables=None,
                                 **kwargs):
            assert fan_out_tuple_factors is not None
            assert req_tables is not None
            # e.g. customer->order
            for r in table.incoming_relationships:
                if r.incoming_table in req_tables and r.outgoing_table in req_tables:
                    continue

                if r not in self.joined_relationships:
                    continue

                if r.outgoing_table.name not in visited_tables:
                    fan_out_tuple_factors.append(r.tf_attribute)
            return False

        custom_bfs(self.requested_tables, process_step=tuple_factor_fan_out,
                   fan_out_tuple_factors=self.fan_out_tuple_factors, req_tables=self.requested_tables)

    def execute(self, current_join):
        start_t = perf_counter()
        input_tuples = len(current_join.df_rows)

        # normalize by weights
        logger.info(f"Normalizing by tuple factors: {self.fan_out_tuple_factors}")
        weights = np.ones(len(current_join.df_rows))
        for tf_fan_out in self.fan_out_tuple_factors:
            assert tf_fan_out.full_name in current_join.df_rows.columns
            current_tfs = current_join.df_rows[tf_fan_out.full_name].values
            assert not np.any(
                np.isnan(current_tfs)), f"Generated tuple factors should not be nan ({tf_fan_out.full_name})"
            assert np.all(current_tfs > 0), f"Generated tuple factors should not be zero ({tf_fan_out.full_name})"
            weights *= current_tfs.reshape(-1)

        assert not np.any(np.isnan(weights)), "Generated tuple factors should not be nan"
        current_join.weights *= 1 / weights
        assert not np.any(np.isnan(current_join.weights)), "Generated tuple factors should not be zero"
        assert not np.any(np.isinf(current_join.weights)), "Generated tuple factors should not be zero"

        # project not-required tuples away
        proj_attributes = []
        for a in current_join.attributes:
            if a.table in self.requested_tables:
                proj_attributes.append(a)
        current_join = current_join.project(proj_attributes)

        self.stats.update({
            'fan_out_tuple_factors': [tf.name for tf in self.fan_out_tuple_factors],
            'input_tuples': input_tuples,
            'elapsed_time': perf_counter() - start_t,
            'step': str(self)
        })

        return current_join

    def step_name(self):
        return f'P{"_".join([t.name for t in self.requested_tables])}'

    def __str__(self):
        return f'ProjectRequestedJoin(all_r={",".join([str(r) for r in self.joined_relationships])}, ' \
               f'tables={",".join([t.name for t in self.requested_tables])})'
