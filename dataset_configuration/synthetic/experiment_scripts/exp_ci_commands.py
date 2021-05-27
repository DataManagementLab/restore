from evaluation.evaluate import EvaluationMethod


def synthetic_confidence_interval_commands(folder='exp_rerr_synthetic_confidence',
                                           evaluation_method=EvaluationMethod.RELATIVE_ERROR):
    """
    Generates commands for Rel. error reduction experiment for the synthetic data.
    :return: list of python command strings
    """

    assert evaluation_method == EvaluationMethod.RELATIVE_ERROR

    experiment_commands = [
        f"""python3 completion.py --preprocess --dataset synthetic 
            --synthetic_no_tuples {synthetic_no_tuples}
            --synthetic_tf_constant {synthetic_tf_constant}
            --synthetic_skew {synthetic_skew}
            --synthetic_correlation {synthetic_correlation}
            --synthetic_no_discrete_values {synthetic_no_discrete_values}
            --normalized_data_directory ../research-data/incomplete-db/synthetic_{percentile}_{predictability_score}_{removal_correlation}_{keep_rate}_{seed}_{synthetic_no_tuples}_{synthetic_tf_constant}_{synthetic_no_discrete_values}_{synthetic_skew}_{synthetic_correlation}/preprocessed
             && 
            python3 completion.py --generate_hdf --dataset synthetic --normalized_data_directory 
            ../research-data/incomplete-db/synthetic_{percentile}_{predictability_score}_{removal_correlation}_{keep_rate}_{seed}_{synthetic_no_tuples}_{synthetic_tf_constant}_{synthetic_no_discrete_values}_{synthetic_skew}_{synthetic_correlation}/preprocessed 
            --hdf_data_directory ../research-data/incomplete-db/synthetic_{percentile}_{predictability_score}_{removal_correlation}_{keep_rate}_{seed}_{synthetic_no_tuples}_{synthetic_tf_constant}_{synthetic_no_discrete_values}_{synthetic_skew}_{synthetic_correlation}/hdf_preprocessed 
             &&
            python3 completion.py --execute_query --scenario_directory ../research-data/incomplete-db/synthetic_{percentile}_{predictability_score}_{removal_correlation}_{keep_rate}_{seed}_{synthetic_no_tuples}_{synthetic_tf_constant}_{synthetic_no_discrete_values}_{synthetic_skew}_{synthetic_correlation}/scenario 
            --dataset synthetic 
            --hdf_data_directory ../research-data/incomplete-db/synthetic_{percentile}_{predictability_score}_{removal_correlation}_{keep_rate}_{seed}_{synthetic_no_tuples}_{synthetic_tf_constant}_{synthetic_no_discrete_values}_{synthetic_skew}_{synthetic_correlation}/hdf_preprocessed
            --projected_tables complete incomplete
            --tf_removals complete.tf_incomplete.complete_id
            --tf_keep_rates 0.5
            --tuple_removal_table incomplete --tuple_removal_keep_rate {keep_rate}
            --removal_attr incomplete.attribute_b
            --removal_method {'bias' if synthetic_skew == '0.0' else 'categorical_prob_bias --removal_attr_values 1'} 
            --removal_attr_bias {removal_correlation} 
            --models {model}
            --model_selection_strategy none --completion_tables incomplete --completable_tables incomplete 
            --seed {seed}
            --target_path experiment_data/{folder}/part_exp_{percentile}_{predictability_score}_{removal_correlation}_{keep_rate}_{seed}_{synthetic_no_tuples}_{synthetic_tf_constant}_{synthetic_no_discrete_values}_{synthetic_skew}_{synthetic_correlation}_{model}.csv
            --synthetic_no_tuples {synthetic_no_tuples}
            --synthetic_tf_constant {synthetic_tf_constant}
            --synthetic_skew {synthetic_skew}
            --synthetic_correlation {synthetic_correlation}
            --synthetic_no_discrete_values {synthetic_no_discrete_values}
            --percentile {percentile}
            --predictability_score {predictability_score}
            --skip_save
            [device_placeholder]
             && 
            rm -r ../research-data/incomplete-db/synthetic_{percentile}_{predictability_score}_{removal_correlation}_{keep_rate}_{seed}_{synthetic_no_tuples}_{synthetic_tf_constant}_{synthetic_no_discrete_values}_{synthetic_skew}_{synthetic_correlation}
            """.replace('\n', ' ')
        for synthetic_no_tuples in ['100000']
        for synthetic_tf_constant in ['5']
        for synthetic_no_discrete_values in ['20']
        for synthetic_skew in ['1.0']
        for synthetic_correlation in ['0.2', '0.4', '0.6', '0.8', '1.0']
        for keep_rate in ['0.2', '0.4', '0.6', '0.8']
        for removal_correlation in ['0.2', '0.4', '0.6', '0.8']
        for percentile in ['0.0', '2.5', '97.5', '100']
        for predictability_score in ['prior_val', 'kl_div_prior']
        for model in ['ar_ep20']
        for seed in ['0', '1', '2', '3', '4']
    ]

    return experiment_commands
