import logging
from copy import copy
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class Aggregation(Enum):
    COUNT = 'count'
    AVG = 'avg'
    SUM = 'sum'

    def __str__(self):
        return self.value


class Operator(Enum):
    GE = '>='
    EQ = '='
    LE = '<='

    def __str__(self):
        return self.value


class AQPQuery:
    def __init__(self, aggregation_attribute=None, where_conditions=None, grouping_attributes=None,
                 completion_tables=None):
        self.aggregation_attribute = aggregation_attribute

        self.where_conditions = where_conditions
        if where_conditions is None:
            self.where_conditions = []

        self.grouping_attributes = grouping_attributes
        if grouping_attributes is None:
            self.grouping_attributes = []

        if completion_tables is None:
            self.completion_tables = []
        else:
            self.completion_tables = set(completion_tables)

    def sql_string(self, aggregation, cat_value_dict=None):
        if cat_value_dict is None:
            cat_value_dict = dict()

        sql_string_components = ["SELECT"]
        if aggregation == Aggregation.COUNT:
            sql_string_components.append("COUNT(*)")
        else:
            sql_string_components.append(f"{str(aggregation).upper()}({self.aggregation_attribute})")

        if len(self.where_conditions) > 0:
            sql_string_components.append("WHERE")
            where_conds = []
            for attr, op, lit in self.where_conditions:
                if attr in cat_value_dict.keys():
                    lit_replace = f"'{cat_value_dict[attr][int(lit)]}'"
                else:
                    lit_replace = lit

                where_conds.append(f"{attr}{str(op)}{lit_replace}")

            sql_string_components.append(" AND ".join(where_conds))

        if len(self.grouping_attributes) > 0:
            sql_string_components.append("GROUP BY")
            sql_string_components.append(", ".join(self.grouping_attributes))

        sql_string = " ".join(sql_string_components)
        sql_string += ";"
        return sql_string

    def compute(self, df, weights=None, upscale=1.0):
        df = copy(df)

        df['weights'] = 1
        if weights is not None:
            df['weights'] = weights

        for attribute, op, literal in self.where_conditions:
            if op == Operator.EQ:
                df = df[df[attribute] == literal]
            elif op == Operator.GE:
                df = df[df[attribute] >= literal]
            elif op == Operator.LE:
                df = df[df[attribute] <= literal]

        counts = None
        averages = None

        if len(df) == 0:
            return 0, None

        if len(self.grouping_attributes) > 0:
            # Count per Group
            df_count = df[self.grouping_attributes + ['weights']].groupby(self.grouping_attributes).sum().reset_index()
            df_count.iloc[:, -1] *= upscale
            counts = [tuple(x) for x in df_count.to_numpy()]

            if self.aggregation_attribute is not None:
                df_agg = df[~df[self.aggregation_attribute].isna()]
                for agg in self.grouping_attributes:
                    df_agg = df[~df[agg].isna()]

                if len(df_agg) > 0:
                    df_agg['weighted_aggregate'] = df_agg['weights'] * df_agg[self.aggregation_attribute]
                    df_agg = df_agg[self.grouping_attributes + ['weighted_aggregate', 'weights']].groupby(
                        self.grouping_attributes).sum()
                    df_agg['weighted_aggregate'] /= df_agg['weights']
                    df_agg = df_agg.reset_index()
                    df_agg.drop(columns=['weights'], inplace=True)
                    averages = [tuple(x) for x in df_agg.to_numpy()]

        else:
            counts = df['weights'].sum() * upscale

            if self.aggregation_attribute is not None:
                df_agg = df[~df[self.aggregation_attribute].isna()]
                for agg in self.grouping_attributes:
                    df_agg = df[~df[agg].isna()]

                if len(df_agg) > 0:
                    df_agg['weighted_aggregate'] = df_agg['weights'] * df_agg[self.aggregation_attribute]
                    averages = df_agg['weighted_aggregate'].sum() / df_agg['weights'].sum()

        return counts, averages


def evaluate_aqp_reduction(q, S, w, removal_attr, aqp_queries, restrict_connected=True,
                           restrict_completion_path=None):
    baseline = q.incomplete_baseline(restrict_connected=restrict_connected,
                                     restrict_completion_path=restrict_completion_path).df_rows
    ground_truth = q.ground_truth(restrict_connected=restrict_connected,
                                  restrict_completion_path=restrict_completion_path).df_rows

    aqp_results = []
    queries = aqp_queries[removal_attr]
    for i, aqp_q in queries.items():
        if aqp_q.completion_tables != q.completion_tables:
            logger.info(f"Query {i} skipped")
            continue

        upscale_factor = 1.0
        if len(baseline) > np.sum(w):
            upscale_factor = len(baseline) / np.sum(w)

        pred = aqp_q.compute(S, weights=w, upscale=upscale_factor)
        pred_baseline = aqp_q.compute(baseline)
        aqp_ground_truth = aqp_q.compute(ground_truth)
        aqp_results.append((i, pred, pred_baseline, aqp_ground_truth))

        logger.info(f"Query {i} evaluated")

    return aqp_results
