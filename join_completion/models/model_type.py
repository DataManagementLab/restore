from enum import Enum


class ModelType(Enum):
    AUTOREGRESSIVE = 'autoregressive'
    SCHEMA_STRUCTURED_AR = 'ssar'

    def __str__(self):
        return self.value
