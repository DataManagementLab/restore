from enum import Enum

from evaluation.approximate_query_processing.aqp_queries import evaluate_aqp_reduction
from evaluation.relative_error.rerr_reduction import evaluate_rerr_reduction


class EvaluationMethod(Enum):
    RELATIVE_ERROR = 'relative_error'
    AQP = 'aqp'

    def __str__(self):
        return self.value


def evaluate(removal_method, removal_attr, removal_attr_values, fixed_completion_path, q, S, w, evaluation_method,
             aqp_queries, schema, tuple_removal_tables):
    evaluation = []
    for removal_method, removal_attr, removal_value, tuple_removal_table in zip(removal_method, removal_attr,
                                                                                removal_attr_values,
                                                                                tuple_removal_tables):
        if removal_attr == 'none' or (S is not None and removal_attr not in S.columns):
            continue

        # evaluate once ignoring all tuples which have no partners at all in the schema
        eval = [False, removal_method, removal_attr]
        if evaluation_method == EvaluationMethod.RELATIVE_ERROR:
            _pred_baseline, _pred_mean, _actual_mean, _actual_no_tuples, _baseline_no_tuples, _sum_tuples = \
                evaluate_rerr_reduction(q, S, w, removal_method, removal_attr, [removal_value])
            eval += [_pred_baseline, _pred_mean, _actual_mean, _actual_no_tuples, _baseline_no_tuples, _sum_tuples]
        elif evaluation_method == EvaluationMethod.AQP:
            eval.append(evaluate_aqp_reduction(q, S, w, removal_attr, aqp_queries))

        evaluation.append(tuple(eval))

        # evaluate once ignoring all tuples which cannot be recovered given the fixed path
        if fixed_completion_path is not None:
            eval = [True, removal_method, removal_attr]
            if evaluation_method == EvaluationMethod.RELATIVE_ERROR:
                _pred_baseline, _pred_mean, _actual_mean, _actual_no_tuples, _baseline_no_tuples, _sum_tuples = \
                    evaluate_rerr_reduction(q, S, w, removal_method, removal_attr, [removal_value],
                                            restrict_connected=False, restrict_completion_path=fixed_completion_path)
                eval += [_pred_baseline, _pred_mean, _actual_mean, _actual_no_tuples, _baseline_no_tuples, _sum_tuples]
            elif evaluation_method == EvaluationMethod.AQP:
                eval.append(evaluate_aqp_reduction(q, S, w, removal_attr, aqp_queries, restrict_connected=False,
                                                   restrict_completion_path=fixed_completion_path))

            evaluation.append(tuple(eval))
    return evaluation
