from enum import Enum


class RemovalMethod(Enum):
    UNIFORM = 'uniform'
    BIAS = 'bias'
    CATEGORICAL_PROB_BIAS = 'categorical_prob_bias'

    def __str__(self):
        return self.value