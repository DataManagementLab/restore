from ast import literal_eval
from enum import Enum

import numpy as np

from evaluation.relative_error.rerr_reduction import rel_err_reduction


class Metric(Enum):
    # Relative Error Reduction
    RERR_RED = 'nae_reduction'
    PRED_BASELINE = 'pred_baseline'
    PRED_MEAN = 'pred_mean'
    ACTUAL_MEAN = 'actual_mean'
    RERR_RED_TUPLES = 'nae_t_reduction'
    SUM_TUPLES = 'sum_tuples'
    ACTUAL_NO_TUPLES = 'actual_no_tuples'
    BASELINE_NO_TUPLES = 'baseline_no_tuples'

    # Query Processing
    QP_IMPROVEMENT = 'qp_improvement'

    def __str__(self):
        return self.value


def literal_eval_second_list(eval_str):
    eval_str = eval_str.replace("<RemovalMethod.CATEGORICAL_PROB_BIAS: 'categorical_prob_bias'>",
                                "'categorical_prob_bias'")
    eval_str = eval_str.replace("<RemovalMethod.BIAS: 'bias'>", "'bias'")

    eval_list = literal_eval(eval_str)

    if len(eval_list) == 2:
        eval_list = [ev for ev in eval_list if ev[0] == True]

    return eval_list[0]


def relative_error(true, predicted, debug=False):
    true = float(true)
    predicted = float(predicted)
    if true == 0:
        return np.nan
    relative_error = (true - predicted) / true
    if debug:
        print(f"\t\tpredicted     : {predicted:.2f}")
        print(f"\t\ttrue          : {true:.2f}")
        print(f"\t\trelative_error: {100 * relative_error:.2f}%")
    return abs(relative_error)


def avg_relative_error(r_dict, r_dict_ground_truth):
    avg_rel_errors = []
    for projection in r_dict_ground_truth.keys():
        predicted = r_dict.get(projection)
        true = r_dict_ground_truth[projection]
        if predicted is None:
            continue
        avg_rel_errors.append(relative_error(true, predicted))

    return np.average(avg_rel_errors)


def result_dict(result_tuples):
    return {t[:-1]: t[-1] for t in result_tuples}


def sum_dict(avg_tuples, count_tuples):
    return {t_avg[:-1]: t_avg[-1] * t_count[-1] for t_avg, t_count in zip(avg_tuples, count_tuples)}


def result_dicts(count, avg):
    r_count = result_dict(count)
    r_avg = None
    r_sum = None

    if avg is not None:
        r_avg = result_dict(avg)
        r_sum = sum_dict(avg, count)

    return r_count, r_avg, r_sum


def avg_errors(baseline, pred, ground_truth):
    avg_improvement = None
    sum_improvement = None

    if isinstance(pred[0], list):
        b_count, b_avg, b_sum = result_dicts(*baseline)
        r_count, r_avg, r_sum = result_dicts(*pred)
        gt_count, gt_avg, gt_sum = result_dicts(*ground_truth)

        count_improvement = avg_relative_error(b_count, gt_count) - avg_relative_error(r_count, gt_count)
        if r_avg is not None:
            avg_improvement = avg_relative_error(b_avg, gt_avg) - avg_relative_error(r_avg, gt_avg)
            sum_improvement = avg_relative_error(b_sum, gt_sum) - avg_relative_error(r_sum, gt_sum)

    else:
        b_count, b_avg = baseline
        r_count, r_avg = pred
        gt_count, gt_avg = ground_truth

        count_improvement = relative_error(b_count, gt_count) - relative_error(r_count, gt_count)

        if r_avg is not None:
            b_sum = b_count * b_avg
            r_sum = r_count * r_avg
            gt_sum = gt_count * gt_avg

            avg_improvement = relative_error(b_avg, gt_avg) - relative_error(r_avg, gt_avg)
            sum_improvement = relative_error(b_sum, gt_sum) - relative_error(r_sum, gt_sum)

    return max(count_improvement, 0), avg_improvement, sum_improvement


def eval_column(row, metric=None):
    evaluation_method = row['evaluation_method']
    eval_list = literal_eval_second_list(row['evaluation'])

    if evaluation_method == 'relative_error':
        fp_opt, removal_method, removal_attr, _pred_baseline, _pred_mean, _actual_mean, _actual_no_tuples, _baseline_no_tuples, _sum_tuples = eval_list
        if metric == Metric.RERR_RED:
            return rel_err_reduction(_pred_baseline, _pred_mean, _actual_mean) * 100
        elif metric == Metric.PRED_BASELINE:
            return _pred_baseline
        elif metric == Metric.PRED_MEAN:
            return _pred_mean
        elif metric == Metric.ACTUAL_MEAN:
            return _actual_mean
        elif metric == Metric.RERR_RED_TUPLES:
            # we know that there are at least the number of tuples we already have
            return max(rel_err_reduction(_baseline_no_tuples, _sum_tuples, _actual_no_tuples) * 100, 0)
        elif metric == Metric.SUM_TUPLES:
            return _sum_tuples
        elif metric == Metric.ACTUAL_NO_TUPLES:
            return _actual_no_tuples
        elif metric == Metric.BASELINE_NO_TUPLES:
            return _baseline_no_tuples

    elif evaluation_method == 'aqp':
        results = dict()

        for query_id, pred, baseline, ground_truth in eval_list[-1]:
            results[query_id] = avg_errors(baseline, pred, ground_truth)

        return results

    return None
