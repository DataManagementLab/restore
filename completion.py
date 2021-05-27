import argparse
import collections
import functools
import logging
import os
import random
import sys

import numpy as np
import torch

from dataset_configuration.airbnb.experiment_scripts.aqp_queries import airbnb_aqp_queries
from dataset_configuration.airbnb.normalization.normalization import normalize_airbnb_schema
from dataset_configuration.airbnb.schema.normalized_schema import gen_normalized_airbnb_schema
from dataset_configuration.imdb.experiment_scripts.aqp_queries import imdb_aqp_queries
from dataset_configuration.imdb.preprocessing.preprocessing import preprocess_imdb_schema
from dataset_configuration.imdb.schema.processed_schema import gen_processed_imdb_schema
from dataset_configuration.synthetic.generation.generate_data import generate_synthetic_data
from dataset_configuration.synthetic.schema.syn_schema import gen_synthetic_schema
from evaluation.evaluate import EvaluationMethod, evaluate
from join_completion.learning.learning import learn_models
from join_completion.learning.params import flat_ar_params, ssar_params
from join_completion.query_compilation.execution import execute_plan
from join_completion.query_compilation.operators.top_path_union import UnionStrategy
from join_completion.query_compilation.planning import incomplete_join_plan, SelectionStrategy, PercentileAttribute, \
    PredictabilityScore
from join_completion.query_compilation.query import Query
from schema_setup.data_preparation.prepare_single_tables import prepare_all_tables
from schema_setup.data_preparation.utils import save_csv
from schema_setup.incomplete_schema_setup.incomplete_schema_generation import generate_incomplete_schema
from schema_setup.incomplete_schema_setup.removal_method import RemovalMethod
from schema_setup.schema.schema_utils import validate_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--dataset', default='airbnb', choices=['airbnb', 'imdb', 'synthetic'])
    parser.add_argument('--raw_data_directory')
    parser.add_argument('--normalized_data_directory')

    parser.add_argument('--generate_hdf', action='store_true')
    parser.add_argument('--hdf_data_directory')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--tuple_removal_table', nargs='+', default='movie')
    parser.add_argument('--tuple_removal_keep_rate', nargs='+', default=1.0, type=float)

    parser.add_argument('--tf_removals', type=str, nargs='+', default=[])
    parser.add_argument('--tf_keep_rates', type=float, nargs='+', default=[])

    # attribute_bias removal
    parser.add_argument('--removal_method', nargs='+', type=RemovalMethod, choices=list(RemovalMethod),
                        default=[RemovalMethod.UNIFORM])
    parser.add_argument('--removal_attr', nargs='+', type=str)
    parser.add_argument('--removal_attr_values', nargs='+', type=str)
    parser.add_argument('--removal_attr_bias', nargs='+', type=float,
                        help='The higher value of attribute the more likely the tuple is kept (for positive '
                             'correlation)')

    parser.add_argument('--projected_tables', type=str, nargs='+')
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--scenario_directory', type=str)
    parser.add_argument('--execute_query', action='store_true')
    parser.add_argument('--no_cascading_deletes', action='store_true')
    parser.add_argument('--models', type=str, nargs='+', default='ar_ep20')
    parser.add_argument('--completion_tables', type=str, nargs='+', default='movie')
    parser.add_argument('--target_path', default=None, type=str)
    parser.add_argument('--skip_save', action='store_true')
    parser.add_argument('--not_synthetic', action='store_true')
    parser.add_argument('--ann_batch_size', default=1000, type=int)
    parser.add_argument('--ann_neighbors_considered', default=10000, type=int)

    parser.add_argument('--force_gen_hdf', action='store_true')
    parser.add_argument('--force_no_baseline', action='store_true')
    parser.add_argument('--max_samples', default=None, type=int)

    parser.add_argument('--evaluation_method', type=EvaluationMethod, choices=list(EvaluationMethod),
                        default=EvaluationMethod.RELATIVE_ERROR)

    parser.add_argument('--model_selection_strategy', default=SelectionStrategy.NONE, type=SelectionStrategy,
                        choices=list(SelectionStrategy))
    parser.add_argument('--top_path_union_strategy', default=UnionStrategy.COMBINE, type=UnionStrategy,
                        choices=list(UnionStrategy))

    parser.add_argument('--validation_tuple_removal_keep_rate', nargs='+', default=[0.4], type=float)
    parser.add_argument('--validation_removal_attr_bias', nargs='+', default=[0.4], type=float)
    parser.add_argument('--validation_tf_keep_rate', default=1.0, type=float)
    parser.add_argument('--completable_tables', default=None, type=str, nargs='+',
                        help='Specify for which incomplete tables we want to be able to run a completion. If given, '
                             'fewer models have to be learned. If None, we assume that every completion can be '
                             'requested and learn all possibly required models.')
    parser.add_argument('--fixed_completion_path', default=None, type=str, nargs='+',
                        help='Path to complete the given query expressed by tables that should be in this path. If not'
                             'specified, model selection will also select a path.')
    # synthetic data
    parser.add_argument('--synthetic_correlation', default=0.0, type=float)
    parser.add_argument('--synthetic_fanout_correlation', default=0.0, type=float)
    parser.add_argument('--synthetic_skew', default=1.0, type=float)
    parser.add_argument('--synthetic_no_tuples', default=1000, type=int)
    parser.add_argument('--synthetic_tf_constant', default=5, type=int)
    parser.add_argument('--synthetic_no_discrete_values', default=1000, type=int)
    parser.add_argument('--force_path_selection', action='store_true')
    parser.add_argument('--percentile', default=None, type=float)
    parser.add_argument('--predictability_score', type=PredictabilityScore, choices=list(PredictabilityScore),
                        default=PredictabilityScore.PRIOR_VAL)

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(__name__)

    # for torch
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(device)
    if args.removal_attr_values is None:
        args.removal_attr_values = ['none'] * len(args.removal_method)

    DatasetConfig = collections.namedtuple('DatasetConfig', 'preprocess gen_schema aqp_queries')
    if args.dataset == 'airbnb':
        dataset_config = DatasetConfig(preprocess=normalize_airbnb_schema,
                                       gen_schema=gen_normalized_airbnb_schema,
                                       aqp_queries=airbnb_aqp_queries)
    elif args.dataset == 'imdb':
        dataset_config = DatasetConfig(preprocess=preprocess_imdb_schema,
                                       gen_schema=gen_processed_imdb_schema,
                                       aqp_queries=imdb_aqp_queries)
    elif args.dataset == 'synthetic':
        dataset_config = DatasetConfig(preprocess=functools.partial(generate_synthetic_data,
                                                                    no_tuples=args.synthetic_no_tuples,
                                                                    tf_constant=args.synthetic_tf_constant,
                                                                    skew=args.synthetic_skew,
                                                                    correlation=args.synthetic_correlation,
                                                                    fanout_correlation=args.synthetic_fanout_correlation,
                                                                    discrete_steps=args.synthetic_no_discrete_values,
                                                                    seed=args.seed),
                                       gen_schema=functools.partial(gen_synthetic_schema,
                                                                    no_tuples=args.synthetic_no_tuples,
                                                                    tf_constant=args.synthetic_tf_constant),
                                       aqp_queries=dict())

    if args.preprocess:
        dataset_config.preprocess(args.raw_data_directory, args.normalized_data_directory)

    if args.generate_hdf:
        prepare_all_tables(dataset_config.gen_schema(), args.normalized_data_directory, args.hdf_data_directory,
                           replace_nans=False, force=args.force_gen_hdf)

    if args.execute_query:

        if args.target_path is None or not os.path.exists(args.target_path):
            schema = dataset_config.gen_schema()

            scenario_name, schema = generate_incomplete_schema(args.scenario_directory, args.dataset, schema,
                                                               args.projected_tables, args.hdf_data_directory,
                                                               args.tf_removals, args.tf_keep_rates,
                                                               args.tuple_removal_table, args.tuple_removal_keep_rate,
                                                               args.removal_method, args.removal_attr,
                                                               args.removal_attr_values, args.removal_attr_bias,
                                                               seed=args.seed,
                                                               cascading_deletes=not args.no_cascading_deletes,
                                                               skip_save=args.skip_save)

            # make sure referenced tables exist etc.
            validate_args(schema, args)

            q = Query(schema, args.completion_tables)

            # training
            model_params = [flat_ar_params(description='ar_test', device=device, layers=1, epochs=1),
                            flat_ar_params(description='ar_ep30', device=device, epochs=30),
                            flat_ar_params(description='ar_ep20', device=device, epochs=20, batches_per_epoch=800),
                            flat_ar_params(description='ar_ep1', device=device, epochs=1, batches_per_epoch=800),
                            ssar_params(description='ssar_test', device=device, epochs=1, layers=1,
                                        layers_hierarchy=1, average_embd=True, hierarchy_embedding_layer_idxs=None),
                            ssar_params(description='ssar_ep30', device=device, epochs=30, max_hierarchy_tuples=10,
                                        max_hierarchy_depth=2),
                            ssar_params(description='ssar_ep20_1st', device=device, epochs=20, max_hierarchy_tuples=10,
                                        hierarchy_embedding_layer_idxs=[0], max_hierarchy_depth=2,
                                        batches_per_epoch=800)
                            ]
            # filter to the params considered due to args
            model_params = [p for p in model_params if p['description'] in args.models]
            assert len(model_params) > 0, "No model selected for learning"
            model_families, training_times, model_accuracies = learn_models(model_params, args.scenario_directory,
                                                                            scenario_name, schema,
                                                                            completable_tables=args.completable_tables,
                                                                            fixed_completion_path=args.fixed_completion_path)
            logger.info(f"Training Times: {training_times}")

            # completion query
            assert len(args.completion_tables)
            # suppress_nan should be False at all times
            # regarding percentile attributes: for this attribute, we will compute the confidence intervals. In case,
            # we had a query like SUM(X) WHERE a=Y, we would have to pass more PercentileAttributes here. (Also tuple
            # factors, this list could be extended in incomplete_join_plan)
            query_plan, prefer_baseline, model_sel_stats = \
                incomplete_join_plan(schema, model_families, q, args.model_selection_strategy,
                                     max_samples=args.max_samples, fully_synthetic=not args.not_synthetic,
                                     validation_removal_attr=args.removal_attr,
                                     validation_removal_attr_values=args.removal_attr_values,
                                     validation_removal_method=args.removal_method,
                                     validation_tuple_removal_table=args.tuple_removal_table,
                                     validation_tuple_removal_keep_rate=args.validation_tuple_removal_keep_rate,
                                     validation_removal_attr_bias=args.validation_removal_attr_bias,
                                     validation_tf_keep_rate=args.validation_tf_keep_rate,
                                     ann_batch_size=args.ann_batch_size,
                                     ann_neighbors_considered=args.ann_neighbors_considered,
                                     fixed_completion_path=args.fixed_completion_path,
                                     top_path_union_strategy=args.top_path_union_strategy,
                                     force_no_baseline=args.force_no_baseline,
                                     force_path_selection=args.force_path_selection,
                                     percentile=args.percentile,
                                     percentile_attributes=[PercentileAttribute(args.removal_attr[0],
                                                                                [args.removal_attr_values[0]],
                                                                                inverted=False)],
                                     predictability_score=args.predictability_score)
            if prefer_baseline and not args.force_path_selection:
                logger.debug("Skipping query execution since baseline was selected by query planning")
                execution_time = 0
                stats = None
                evaluation = evaluate(args.removal_method, args.removal_attr, args.removal_attr_values,
                                      args.fixed_completion_path, q, None, None, args.evaluation_method,
                                      dataset_config.aqp_queries, schema, args.tuple_removal_table)

            else:
                S, w, execution_time, stats = execute_plan(query_plan,
                                                           os.path.join(args.scenario_directory, scenario_name),
                                                           skip_save=args.skip_save)
                logger.info(stats)

                # evaluation
                evaluation = evaluate(args.removal_method, args.removal_attr, args.removal_attr_values,
                                      args.fixed_completion_path, q, S, w, args.evaluation_method,
                                      dataset_config.aqp_queries, schema, args.tuple_removal_table)

            # save all parameters
            if args.target_path is not None:
                csv_rows = [{
                    'dataset': args.dataset,
                    'models': str(args.models),
                    'projected_tables': str(args.projected_tables),
                    'completion_tables': str(args.completion_tables),
                    'tf_removals': str(args.tf_removals),
                    'tf_keep_rates': str(args.tf_keep_rates),
                    'tuple_removal_table': str(args.tuple_removal_table),
                    'tuple_removal_keep_rate': str(args.tuple_removal_keep_rate),
                    'removal_method': str([str(rm) for rm in args.removal_method]),
                    'removal_attr': str(args.removal_attr),
                    'removal_attr_values': str(args.removal_attr_values),
                    'removal_attr_bias': str(args.removal_attr_bias),
                    'seed': str(args.seed),
                    'cascading_deletes': not args.no_cascading_deletes,
                    'model_families': str([str(mf) for mf in model_families]),
                    'training_times': str(training_times),
                    'query_plan': str(query_plan),
                    'execution_time': execution_time,
                    'evaluation': evaluation,
                    'fully_synthetic': not args.not_synthetic,
                    'model_selection_strategy': args.model_selection_strategy,
                    'validation_tuple_removal_keep_rate': args.validation_tuple_removal_keep_rate,
                    'validation_removal_attr_bias': args.validation_removal_attr_bias,
                    'validation_tf_keep_rate': args.validation_tf_keep_rate,
                    'not_synthetic': args.not_synthetic,
                    'ann_batch_size': args.ann_batch_size,
                    'ann_neighbors_considered': args.ann_neighbors_considered,
                    'completable_tables': args.completable_tables,
                    'fixed_completion_path': args.fixed_completion_path,
                    'execution_stats': stats,
                    'top_path_union_strategy': args.top_path_union_strategy,
                    'prefer_baseline': prefer_baseline,
                    'code_comment': 'Now numba discretizing & efficient ANN (no repeat)',
                    'evaluation_method': args.evaluation_method,
                    'synthetic_correlation': str(args.synthetic_correlation),
                    'synthetic_fanout_correlation': str(args.synthetic_fanout_correlation),
                    'synthetic_skew': str(args.synthetic_skew),
                    'synthetic_no_tuples': args.synthetic_no_tuples,
                    'synthetic_tf_constant': args.synthetic_tf_constant,
                    'synthetic_no_discrete_values': args.synthetic_no_discrete_values,
                    'model_sel_stats': model_sel_stats,
                    'model_accuracies': model_accuracies,
                    'percentile': args.percentile,
                    'predictability_score': args.predictability_score
                }]
                save_csv(csv_rows, args.target_path)
