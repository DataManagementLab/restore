from dataset_configuration.airbnb.experiment_scripts.exp_commands import airbnb_commands
from dataset_configuration.imdb.experiment_scripts.exp_commands import imdb_commands
from dataset_configuration.synthetic.experiment_scripts.exp_ci_commands import synthetic_confidence_interval_commands
from dataset_configuration.synthetic.experiment_scripts.exp_commands import synthetic_commands
from evaluation.evaluate import EvaluationMethod


def gen_experiment_commands():
    experiment_commands = []
    for syn in [True, False]:
        for model_search in [True, False]:
            if syn and model_search:
                continue

            for evaluation_method in [EvaluationMethod.RELATIVE_ERROR, EvaluationMethod.AQP]:
                experiment_commands += imdb_commands(synthetic=syn, model_search=model_search,
                                                     evaluation_method=evaluation_method,
                                                     folder=f'exp_{str(evaluation_method)}_imdb')
                experiment_commands += airbnb_commands(synthetic=syn, model_search=model_search,
                                                       evaluation_method=evaluation_method,
                                                       folder=f'exp_{str(evaluation_method)}_airbnb')
    experiment_commands += synthetic_commands(folder=f'exp_{str(EvaluationMethod.RELATIVE_ERROR)}_synthetic')
    experiment_commands += synthetic_confidence_interval_commands(folder='exp_rerr_synthetic_confidence',
                                                                  evaluation_method=EvaluationMethod.RELATIVE_ERROR)

    assert len(set(experiment_commands)) == len(experiment_commands)
    print(f'Generated {len(experiment_commands)} experiment commands')

    return experiment_commands


if __name__ == '__main__':
    for cmd in gen_experiment_commands():
        # replace with --device cuda for GPU
        print(cmd.replace('[device_placeholder]', ''))
