import logging
from enum import Enum

from join_completion.query_compilation.operators.plan import Plan
from schema_setup.schema.schema_utils import stable_hash
import numpy as np

logger = logging.getLogger(__name__)


class UnionStrategy(Enum):
    COMBINE = 'combine'
    SCALE_UP = 'scale_up'

    def __str__(self):
        return self.value


class TopPathUnion(Plan):
    """
    Combines several completion paths.
    """

    def __init__(self, physical_plans, final_relationships, top_path_union_strategy):
        Plan.__init__(self)
        self.physical_plans = physical_plans
        self.final_relationships = final_relationships
        self.top_path_union_strategy = top_path_union_strategy
        assert len(physical_plans) == len(final_relationships)

    def execute(self):
        project_zero = []
        S_all = None
        stats = []

        base_size = None
        estimated_size = None

        for plan, final_r in zip(self.physical_plans, self.final_relationships):
            logger.info(f"Executing subplan")
            for s in plan:
                logger.info(f"\t {str(s)}")
            S_i = None
            for step in plan:
                S_i = step.execute(S_i)
            stats.append([step.stats for step in plan])

            # only consider tuples we have not yet received using another completion path
            # i.e., restrict to tuples where corresponding tfs are zero
            len_before = len(S_i.df_rows)
            for tf_z in project_zero:
                interesting_idx = (S_i.df_rows[tf_z.full_name] == 0)
                S_i.df_rows = S_i.df_rows[interesting_idx]
                S_i.weights = S_i.weights[interesting_idx]
            logger.info(f"Filtering additional tuples ({'='.join([tf_z.full_name for tf_z in project_zero])}=0)")
            logger.info(f"\t Remaining: {len(S_i.df_rows) / len_before * 100:.2f}% (i.e., {len(S_i.df_rows)} rows, "
                        f"{np.sum(S_i.weights):.0f} exp. tuples)")
            project_zero.append(final_r.tf_attribute)

            if self.top_path_union_strategy == UnionStrategy.COMBINE:

                # combine with previously completed tuples
                if S_all is None:
                    S_all = S_i
                else:
                    S_all = S_all.concat(S_i)

            elif self.top_path_union_strategy == UnionStrategy.SCALE_UP:
                if S_all is None:
                    S_all = S_i
                    base_size = np.sum(S_i.weights)
                    estimated_size = np.sum(S_i.weights)

                else:
                    estimated_size += np.sum(S_i.weights)

        if self.top_path_union_strategy == UnionStrategy.SCALE_UP:
            S_all.weights *= estimated_size / base_size

        return S_all, stats

    def plan_name(self):
        query_plan_identifier = '_'.join([','.join([str(s) for s in p]) for p in self.physical_plans])
        return f'TopPathUnion({self.top_path_union_strategy},{query_plan_identifier})'

    @property
    def identifier(self):
        return stable_hash(self.plan_name())

    def explain(self):
        logger.info(f"TopPathUnion({self.top_path_union_strategy})")
        for plan, final_r in zip(self.physical_plans, self.final_relationships):
            logger.info(f"Subplan w final relationship {final_r}")
            for s in plan:
                logger.info(f"\t {str(s)}")

    def __str__(self):
        return self.plan_name()
