import logging
from copy import copy

import numpy as np

from schema_setup.incomplete_schema_setup.removal_method import RemovalMethod

logger = logging.getLogger(__name__)


def nan_mean(rows, attribute):
    rows = filter_not_nan(attribute, rows)
    nanmean = rows[attribute.full_name].mean()

    return nanmean


def nan_ratio(rows, attribute, attribute_values):
    rows = filter_not_nan(attribute, rows)
    return len(rows[rows[attribute.full_name].isin(attribute_values)]) / len(rows)


def filter_not_nan(attribute, rows):
    rows = rows.df_rows
    len_before = len(rows)
    rows = rows.loc[~rows[attribute.full_name].isna()]
    logger.info(f"Filtered {len(rows) / len_before * 100:.2f}% not-nan values")
    return rows


def rel_err_reduction(baseline_pred, pred_mean, actual_mean):
    rerr_baseline = baseline_pred - actual_mean
    rerr_predicted = pred_mean - actual_mean
    if rerr_baseline == 0:
        return 0
    return 1 - abs(rerr_predicted / rerr_baseline)


def evaluate_rerr_reduction(q, S, w, removal_method, removal_attr, removal_attr_values, restrict_connected=True,
                            restrict_completion_path=None):
    S, actual_no_tuples, baseline_no_tuples, baseline, ground_truth, sum_tuples, w = preprocess(S, q, removal_attr,
                                                                                                restrict_connected, w,
                                                                                                restrict_completion_path)

    if removal_method == RemovalMethod.BIAS:

        pred_baseline = nan_mean(baseline, baseline.attribute_dict[removal_attr])
        actual_mean = nan_mean(ground_truth, ground_truth.attribute_dict[removal_attr])

        if S is not None:
            pred_mean = np.sum(S[removal_attr] * w) / np.sum(w)
        else:
            pred_mean = pred_baseline

    elif removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:

        removal_attr_values = set(removal_attr_values)

        pred_baseline = nan_ratio(baseline, baseline.attribute_dict[removal_attr], removal_attr_values)
        actual_mean = nan_ratio(ground_truth, ground_truth.attribute_dict[removal_attr], removal_attr_values)

        if S is not None:
            app_idx = S[removal_attr].isin(removal_attr_values)
            # prediction how often the categorical values appear
            pred_mean = np.sum(w[app_idx]) / np.sum(w)
        else:
            pred_mean = pred_baseline

    else:
        raise NotImplementedError

    logger.info(f"Evaluating de-biasing removal_method: {removal_method}, removal_attr: {removal_attr}")
    logger.info(f"\tPredicted (no_tuples): {sum_tuples:.0f}")
    logger.info(f"\tActual    (no_tuples): {actual_no_tuples:.0f}")
    logger.info(f"\tRel. error: {(sum_tuples - actual_no_tuples) / actual_no_tuples * 100:.2f}%")

    logger.info(f"\tPredicted Baseline ({removal_attr}): {pred_baseline:.5f}")
    logger.info(f"\tPredicted          ({removal_attr}): {pred_mean:.5f}")
    logger.info(f"\tActual             ({removal_attr}): {actual_mean:.5f}")
    logger.info(f"\tRel err reduction: {rel_err_reduction(pred_baseline, pred_mean, actual_mean) * 100:.2f}%")
    logger.info(f"\tRel. error    : {(pred_mean - actual_mean) / actual_mean * 100:.2f}%")

    return pred_baseline, pred_mean, actual_mean, actual_no_tuples, baseline_no_tuples, sum_tuples


def preprocess(S, q, removal_attr, restrict_connected, w, restrict_completion_path):
    baseline = q.incomplete_baseline(restrict_connected=restrict_connected,
                                     restrict_completion_path=restrict_completion_path)
    ground_truth = q.ground_truth(restrict_connected=restrict_connected,
                                  restrict_completion_path=restrict_completion_path)
    actual_no_tuples = len(ground_truth.df_rows)
    baseline_no_tuples = len(baseline.df_rows)

    sum_tuples = 0
    if S is not None and w is not None:
        sum_tuples = np.sum(w)
        len_before = len(S)
        rel_idx = ~np.isnan(S[removal_attr])
        S = S[rel_idx]
        w = w[rel_idx]
        logger.info(f"Removing Nans from synthetic data {len_before}->{len(S)}")
    return S, actual_no_tuples, baseline_no_tuples, baseline, ground_truth, sum_tuples, w


def mae(q, S, w, pk_attribute, attribute, validation_removal_method, validation_removal_attr_values,
        restrict_connected=True, restrict_completion_path=None):
    S, actual_no_tuples, baseline_no_tuples, _, ground_truth, sum_tuples, w = preprocess(S, q, attribute,
                                                                                         restrict_connected, w,
                                                                                         restrict_completion_path)

    if validation_removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:

        g_agg = copy(ground_truth.df_rows)
        g_agg['temp'] = 0
        g_agg.loc[g_agg[attribute].isin(validation_removal_attr_values), 'temp'] = 1
        g_agg[attribute] = g_agg['temp']
        g_agg = g_agg.groupby([pk_attribute]).mean().reset_index()

        S['wa'] = 0
        S.loc[S[attribute].isin(validation_removal_attr_values), 'wa'] = 1
        S['wa'] *= w

    else:
        g_agg = ground_truth.df_rows.groupby([pk_attribute]).mean().reset_index()
        S['wa'] = w * S[attribute]
    S['weights'] = w
    S_agg = S[[pk_attribute, 'wa', 'weights']].groupby([pk_attribute]).sum()
    S_agg['wa'] /= S_agg['weights']
    S_agg = S_agg.reset_index()

    S_agg = S_agg.merge(g_agg, left_on=pk_attribute, right_on=pk_attribute)
    S_agg = S_agg[~S_agg[attribute].isna()]

    mae = np.mean(np.abs(S_agg['wa'] - S_agg[attribute]))
    logger.info(f"MAE {mae}")
    return mae
