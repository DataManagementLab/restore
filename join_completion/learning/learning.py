import os


class ModelFamily:
    def __init__(self, params):
        # check if this should be done this way
        self.r_model_dict = dict()
        self.params = params

    def __str__(self):
        return '_'.join([str(m) for m in self.r_model_dict.values()])

    def map_to_validation_schema(self, t_mapping, r_mapping, a_mapping):
        mf_mapped = ModelFamily(self.params)

        mf_mapped.r_model_dict = {(r_mapping[r], inverse): m.map_to_validation_schema(t_mapping, r_mapping, a_mapping)
                                  for (r, inverse), m in self.r_model_dict.items()}
        return mf_mapped


def learn_models(params, scenario_directory, scenario_name, schema, completable_tables=None,
                 fixed_completion_path=None):
    model_families = []
    mf_training_times = []
    mf_accuracies = []
    for p in params:

        # create subfolder for model
        model_directory = os.path.join(scenario_directory, scenario_name, p['model_subfolder'])
        os.makedirs(model_directory, exist_ok=True)
        model_family = ModelFamily(p)

        # find models required for schema
        models = p['find_required_models'](schema, model_directory=model_directory, params=p,
                                           completable_tables=completable_tables,
                                           fixed_completion_path=fixed_completion_path)
        training_times = []
        accuracies = []
        for m in models:
            training_time, accuracy = m.train()
            training_times.append(training_time)
            accuracies.append(accuracy)
            for cs_r in m.completion_relationships:
                model_family.r_model_dict[(cs_r.r, cs_r.inverse)] = m
        model_families.append(model_family)
        mf_training_times.append(training_times)
        mf_accuracies.append(accuracies)

    return model_families, mf_training_times, mf_accuracies
