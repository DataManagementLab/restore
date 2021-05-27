from evaluation.evaluate import EvaluationMethod


def imdb_commands(synthetic=True, model_search=False, folder='exp_rerr_imdb',
                  evaluation_method=EvaluationMethod.RELATIVE_ERROR):
    """
    Generates commands for Rel. error reduction experiment.
    :return: list of python command strings
    """

    experiment_commands = []

    eval_setups, synthetic_conf = determine_setups(evaluation_method, synthetic)

    tables = [('movie', ['movie'], fixed_movie_path), ('director', ['director'], fixed_path_via_movie),
              ('company', ['company'], fixed_path_via_movie)]
    attributes = {
        'movie': [
            ('country_14',
             '--removal_method categorical_prob_bias --removal_attr movie.country --removal_attr_values 14'),
            ('genre_4', '--removal_method categorical_prob_bias --removal_attr movie.genre --removal_attr_values 4'),
            ('production_year', '--removal_method bias --removal_attr movie.production_year'),
        ],
        'director': [
            ('birth_year', '--removal_method bias uniform --removal_attr director.birth_year none'),
        ],
        'company': [
            ('country_code_1',
             '--removal_method categorical_prob_bias uniform --removal_attr company.country_code none --removal_attr_values 1 none')
        ]
    }

    if model_search:
        models = [('all_4', 'ar_ep30 ar_ep20 ssar_ep30 ssar_ep20_1st')]
        # models = [('all_2', 'ar_ep30 ssar_ep30'), ('all_4', 'ar_ep30 ar_ep20 ssar_ep30 ssar_ep20_1st')]
        model_selection_strategies = [('all_art_bias_0.6_0.6',
                                       'artificial_bias --validation_tuple_removal_keep_rate 0.6 --validation_removal_attr_bias 0.6'),
                                      # ('all_art_bias_0.4_0.4',
                                      #  'artificial_bias --validation_tuple_removal_keep_rate 0.4 --validation_removal_attr_bias 0.4')
                                      ]

        setups = {table_name: [('', '')] for table_name, _, _ in tables}
    else:
        models = [('ar_ep30', 'ar_ep30'), ('ar_ep20', 'ar_ep20'), ('ssar_ep30', 'ssar_ep30'),
                  ('ssar_ep20_1st', 'ssar_ep20_1st')]
        model_selection_strategies = [('', 'none')]
        setups = {'movie': [('fp_mov_dir', '--fixed_completion_path director movie_director movie'),
                            ('fp_mov_cp', '--fixed_completion_path company movie_companies movie'),
                            ('fp_mov_ac', '--fixed_completion_path actor movie_actor movie')],
                  'director': [('fp_dir_ac', '--fixed_completion_path actor movie_actor movie movie_director director'),
                               ('fp_dir_cp',
                                '--fixed_completion_path company movie_companies movie movie_director director')],
                  'company': [('fp_cp_ac', '--fixed_completion_path actor movie_actor movie movie_companies company'),
                              ('fp_cp_dir',
                               '--fixed_completion_path director movie_director movie movie_companies company')],
                  'actor': [('fp_ac_dir', '--fixed_completion_path director movie_director movie movie_actor actor'),
                            ('fp_ac_cp', '--fixed_completion_path company movie_companies movie movie_actor actor')]
                  }

    for removal_table, completion_tables, path_function in tables:
        experiment_commands += path_function(attributes[removal_table], setups[removal_table], synthetic_conf,
                                             removal_table,
                                             completion_tables, model_selection_strategies, models, folder, eval_setups)

    experiment_commands += imdb_join_aqp_commands(setups, synthetic_conf, model_selection_strategies, attributes,
                                                  models, folder, eval_setups, evaluation_method)

    return experiment_commands


def imdb_join_aqp_commands(setups, synthetic_conf, model_selection_strategies, attributes, models, folder, eval_setups,
                           evaluation_method):
    experiment_commands = []
    if evaluation_method != EvaluationMethod.AQP:
        return experiment_commands

    tables = [('movie', ['movie', 'movie_companies', 'company'], fixed_movie_path),
              ('movie', ['movie', 'movie_director', 'director'], fixed_movie_path),
              ('director', ['director', 'movie_director', 'movie'], fixed_path_via_movie),
              ('company', ['company', 'movie_companies', 'company'], fixed_path_via_movie)]

    for removal_table, completion_tables, path_function in tables:
        experiment_commands += path_function(attributes[removal_table], setups[removal_table], synthetic_conf,
                                             removal_table,
                                             completion_tables, model_selection_strategies, models, folder, eval_setups)

    return experiment_commands


def determine_setups(evaluation_method, synthetic):
    if synthetic:
        synthetic = [('', '')]
    else:
        synthetic = [('ann_1k', '--not_synthetic --ann_batch_size 1000 --ann_neighbors_considered 10000')]

    if evaluation_method == EvaluationMethod.RELATIVE_ERROR:
        eval_setups = ['']
    elif evaluation_method == EvaluationMethod.AQP:
        eval_setups = ['--evaluation_method aqp']
    return eval_setups, synthetic


def fixed_movie_path(attributes, setups, synthetic, removal_table, completion_tables, model_selection_strategies,
                     models, folder, eval_setups):
    # join makes sure that the different selection strategies are co-located on the same node
    experiment_commands = [
        ' && '.join([f"""python3 completion.py --execute_query 
            --scenario_directory ../research-data/incomplete-db/imdb/scenario    
            --dataset imdb
            --hdf_data_directory ../research-data/incomplete-db/imdb/hdf_preprocessed
            --projected_tables movie movie_companies company movie_director director movie_actor actor
            --tf_removals company.tf_movie_companies.company_id director.tf_movie_director.person_id actor.tf_movie_actor.person_id
            --tf_keep_rates 0.2 0.2 0.2
            --tuple_removal_table {removal_table}
            --tuple_removal_keep_rate {keep_rate}
            {attribute}
            --removal_attr_bias {correlation}
            --models {model} --model_selection_strategy {model_selection_strat}
            --completion_tables {' '.join(completion_tables)}
            --completable_tables {' '.join(completion_tables)}
             {path}
            --seed {seed} {syn} {eval}
            --target_path experiment_data/{folder}/part_exp_{mssid}{syn_id}_{path_id}_{keep_rate}_{attribute_id}{'_'.join(completion_tables) if len(completion_tables) > 1 else ''}_{correlation}_{mid}_{seed}.csv
            --skip_save [device_placeholder]
            """.replace('\n', ' ') for mssid, model_selection_strat in model_selection_strategies])
        for path_id, path in setups
        for syn_id, syn in synthetic
        for eval in eval_setups
        for keep_rate in ['0.2', '0.4', '0.6', '0.8']
        for attribute_id, attribute in attributes
        for correlation in ['0.2', '0.4', '0.6', '0.8']
        for mid, model in models
        for seed in ['0']
    ]
    return experiment_commands


def fixed_path_via_movie(attributes, setups, synthetic, removal_table, completion_tables, model_selection_strategies,
                         models, folder, eval_setups):
    experiment_commands = [
        ' && '.join([f"""python3 completion.py --execute_query 
            --scenario_directory ../research-data/incomplete-db/imdb/scenario    
            --dataset imdb
            --hdf_data_directory ../research-data/incomplete-db/imdb/hdf_preprocessed
            --projected_tables movie movie_companies company movie_director director movie_actor actor
            --tf_removals company.tf_movie_companies.company_id director.tf_movie_director.person_id actor.tf_movie_actor.person_id
            --tf_keep_rates 0.2 0.2 0.2
            --tuple_removal_table {removal_table} movie
            --tuple_removal_keep_rate {keep_rate} 0.8
            {attribute}
            --removal_attr_bias {correlation} 0.0
            --models {model} --model_selection_strategy {model_selection_strat}
            --completion_tables {' '.join(completion_tables)}
            --completable_tables {' '.join(completion_tables)}
             {path}
            --seed {seed} {syn} {eval}
            --target_path experiment_data/{folder}/part_exp_{mssid}{syn_id}_{path_id}_{keep_rate}_{attribute_id}{'_'.join(completion_tables) if len(completion_tables) > 1 else ''}_{correlation}_{mid}_{seed}.csv
            --skip_save [device_placeholder]
            """.replace('\n', ' ') for mssid, model_selection_strat in model_selection_strategies])
        for path_id, path in setups
        for syn_id, syn in synthetic
        for eval in eval_setups
        for keep_rate in ['0.2', '0.4', '0.6', '0.8']
        for attribute_id, attribute in attributes
        for correlation in ['0.2', '0.4', '0.6', '0.8']
        for mid, model in models
        for seed in ['0']
    ]
    return experiment_commands
