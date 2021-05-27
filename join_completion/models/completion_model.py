from schema_setup.schema.schema_utils import stable_hash


class CompletionModel:
    def __init__(self, model_directory):
        self.completion_relationships = set()
        self.model_directory = model_directory
        self.model = None

    def train(self):
        raise NotImplementedError

    def return_parameters(self):
        return {'type': 'CompletionModel',
                'input_encoding': None,
                'output_encoding': None,
                'epochs': None,
                'fit_existing_weights': None,
                'max_weight_distributions': None,
                'epsilon_regularizer': None,
                'sample_posterior_weights': None,
                'sets_sampled': None,
                'min_set_counts': None,
                'max_set_counts': None,
                'impossible_imputation': None,
                'ignore_p': None,
                'max_iterations': None,
                'delta_sampling': None,
                'train_samples': None,
                'train_only_incomplete': None,
                'neg_delta_ignore': None,
                'heads': None,
                'warmups': None,
                'layers': None,
                'layers_set': None,
                'fc_hidden': None,
                'ignore_sets': None,
                'average_embd': None,
                'set_embedding_layer_idxs': None}

    @property
    def model_name(self):
        params = self.return_parameters()
        model_name = str.lower(str(params['type']))

        param_desc = ''
        for key, value in params.items():
            if key in {'type', 'device', 'find_required_models', 'ignore_sets', 'model_subfolder'}:
                continue
            if key == 'self_evidence_only':
                # only add if parameter is true
                if value:
                    param_desc += '_se_'
                continue
            if value is not None:
                param_desc += f'_{key}' + str(value)

        # also add completion relationships to distinguish models with same parameters that were built over different
        # parts of the schema
        cr_desc = sorted([str(cr) for cr in self.completion_relationships])
        param_desc += '_'.join(cr_desc)

        return model_name + stable_hash(param_desc)

    def map_to_validation_schema(self, t_mapping, r_mapping, a_mapping):
        raise NotImplementedError


class CompletionSetup:
    def __init__(self, evidence_relationships, r, completion_relationships, inverse):
        self.evidence_relationships = evidence_relationships
        self.r = r
        self.completion_relationships = completion_relationships
        self.inverse = inverse

    @property
    def evidence_table(self):
        evidence_table = self.r.incoming_table
        if self.inverse:
            evidence_table = self.r.outgoing_table
        return evidence_table

    @property
    def evidence_tables(self):
        evidence_tables = {self.evidence_table}
        for r in self.evidence_relationships:
            evidence_tables.add(r.incoming_table)
            evidence_tables.add(r.outgoing_table)
        return evidence_tables

    @property
    def completion_table(self):
        completion_table = self.r.outgoing_table
        if self.inverse:
            completion_table = self.r.incoming_table
        return completion_table

    @property
    def tables(self):
        return self.evidence_tables.union({self.completion_table})

    def project_evidence_attributes(self, learned_attributes, exclude=None):
        if exclude is None:
            exclude = set()

        ev_attributes = set(self.evidence_table.attributes)
        for r in self.evidence_relationships:
            ev_attributes.update(r.incoming_table.attributes)
            ev_attributes.update(r.outgoing_table.attributes)

        return [a for a in learned_attributes if a in ev_attributes and a not in exclude]

    def __str__(self):
        str_r = 'CompletionSetup('
        str_r += self.evidence_table.name + '->' + self.completion_table.name
        if len(self.evidence_relationships) > 0:
            str_r += ', evidence=' + ','.join([str(r) for r in self.evidence_relationships])
        if len(self.completion_relationships) > 0:
            str_r += ', completion_ev=' + ','.join([str(r) for r in self.completion_relationships])
        str_r += ')'
        return str_r

    def __eq__(self, other):
        if not isinstance(other, CompletionSetup):
            return False

        return hash(self) == hash(other)

    def __hash__(self):
        return hash(
            (frozenset(self.evidence_relationships), self.r, frozenset(self.completion_relationships), self.inverse))
