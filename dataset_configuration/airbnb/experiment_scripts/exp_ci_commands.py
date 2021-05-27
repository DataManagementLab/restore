from dataset_configuration.imdb.experiment_scripts.exp_commands import determine_setups
from evaluation.evaluate import EvaluationMethod


def airbnb_confidence_interval_commands(synthetic=True, folder='exp_relative_error_airbnb',
                                        evaluation_method=EvaluationMethod.RELATIVE_ERROR):
    """
    Generates confidence interval commands for the categorical datasets
    """

    experiment_commands = []

    eval_setups, synthetic_conf = determine_setups(evaluation_method, synthetic)

    tables = [('listings', ['listings'], completion_command),
              ('hosts', ['hosts'], completion_command)
              ]

    attributes = {
        'listings': [
            ('room_type_1',
             '--removal_method categorical_prob_bias --removal_attr listings.room_type --removal_attr_values 1'),
            ('property_type_3',
             '--removal_method categorical_prob_bias --removal_attr listings.property_type --removal_attr_values 3'),
        ],
        'hosts': [
        ],
    }

    models = [('ar_ep30', 'ar_ep30'), ('ar_ep20', 'ar_ep20'), ('ssar_ep30', 'ssar_ep30'),
              ('ssar_ep20_1st', 'ssar_ep20_1st')]
    model_selection_strategies = [('', 'none')]
    setups = {'listings': [('fp_h', '--fixed_completion_path neighborhoods listings'),
                           ('fp_n', '--fixed_completion_path hosts listings')],
              'hosts': [('', '')]
              }

    for removal_table, completion_tables, path_function in tables:
        experiment_commands += path_function(attributes[removal_table], setups[removal_table], synthetic_conf,
                                             removal_table,
                                             completion_tables, model_selection_strategies, models, folder, eval_setups)

    return experiment_commands


def completion_command(attributes, setups, synthetic, removal_table, completion_tables, model_selection_strategies,
                       models, folder, eval_setups):
    # join makes sure that the different selection strategies are co-located on the same node
    experiment_commands = [
        ' && '.join([f"""python3 completion.py --execute_query 
            --scenario_directory ../research-data/incomplete-db/airbnb/scenario    
            --dataset airbnb
            --hdf_data_directory ../research-data/incomplete-db/airbnb/hdf_preprocessed
            --projected_tables neighborhoods hosts listings
            --tf_removals neighborhoods.tf_listings.neighborhood_id hosts.tf_listings.host_id
            --tf_keep_rates 0.3 0.3
            --tuple_removal_table {removal_table}
            --tuple_removal_keep_rate {keep_rate}
            {attribute}
            --removal_attr_bias {correlation}
            --models {model} --model_selection_strategy {model_selection_strat}
            --completion_tables {' '.join(completion_tables)}
            --completable_tables {' '.join(completion_tables)}
             {path}
            --percentile {percentile}
            --predictability_score {predictability_score}
            --seed {seed} {syn} {eval}
            --target_path experiment_data/{folder}/part_syn_exp_{predictability_score}_{percentile}_{mssid}{syn_id}_{path_id}_{keep_rate}_{attribute_id}{'_'.join(completion_tables) if len(completion_tables) > 1 else ''}_{correlation}_{mid}_{seed}.csv
            --skip_save [device_placeholder]
            """.replace('\n', ' ') for mssid, model_selection_strat in model_selection_strategies
                     for percentile in ['2.5', '97.5']
                     for predictability_score in ['kl_div_prior']
                     ])
        for path_id, path in setups
        for syn_id, syn in synthetic
        for eval in eval_setups
        for keep_rate in ['0.2', '0.4', '0.6', '0.8']
        for attribute_id, attribute in attributes
        for correlation in ['0.2', '0.4', '0.6', '0.8']
        for mid, model in models
        for seed in ['0', '1', '2', '3', '4']
        # for seed in ['0']
    ]
    return experiment_commands
