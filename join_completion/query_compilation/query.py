import logging

import numpy as np

from schema_setup.schema.dataset import Dataset
from schema_setup.schema.schema_utils import join_tables

logger = logging.getLogger(__name__)


class Query:
    def __init__(self, schema, completion_tables):
        assert len(completion_tables) > 0

        self.single_target_table = None
        self.target_relationships = None
        self.completion_tables = set(completion_tables)

        if len(completion_tables) == 1:
            self.single_target_table = schema.table_dict[completion_tables[0]]

        else:
            self.target_relationships = {r for r in schema.relationships if
                                         str(r.incoming_table) in completion_tables and
                                         str(r.outgoing_table) in completion_tables}
            assert len(self.target_relationships) > 0, "No relationship found"

    def ground_truth(self, restrict_connected=True, incomplete=False, restrict_completion_path=None):
        if self.target_relationships is not None:
            ground_truth = join_tables(self.target_relationships, incomplete_join=incomplete, keep_tfs=True)
        else:
            if incomplete:
                ground_truth = self.single_target_table.incomplete_dataset_with_tfs
            else:
                ground_truth = self.single_target_table.load()

        len_before = len(ground_truth.df_rows)
        remove_idx = None
        for a in ground_truth.attributes:
            if a.is_tf:
                schema = a.table.schema
                r = [r for r in schema.relationships if r.tf_attribute == a][0]

                if restrict_connected or (restrict_completion_path is not None and
                                          r.tables.issubset({schema.table_dict[t] for t in restrict_completion_path})):

                    assert not np.any(np.isnan(ground_truth.df_rows[a.full_name].values))
                    if remove_idx is None:
                        remove_idx = ground_truth.df_rows[a.full_name] == 0
                    else:
                        remove_idx &= ground_truth.df_rows[a.full_name] == 0

        if remove_idx is not None:
            ground_truth = Dataset(ground_truth.df_rows[~remove_idx], ground_truth.attributes)
            logger.warning("Only considering tuples with join partners as ground truth")
            logger.warning(f"This reduced the number of tuples from {len_before} to {len(ground_truth.df_rows)} "
                           f"({len(ground_truth.df_rows) / len_before * 100:.2f}% left)")

        return ground_truth

    def incomplete_baseline(self, restrict_connected=True, restrict_completion_path=None):
        return self.ground_truth(restrict_connected=restrict_connected, incomplete=True,
                                 restrict_completion_path=restrict_completion_path)
