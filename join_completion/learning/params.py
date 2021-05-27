from join_completion.learning.flat_ar import find_flat_ar_models, find_ssar_models
from join_completion.models.model_type import ModelType


def flat_ar_params(**kwargs):
    p = {'input_encoding': 'embed',
         'output_encoding': 'embed',
         'epochs': 20,
         'batches_per_epoch': 500,
         'warmups': 8000,
         'layers': 5,
         'fc_hidden': 256,
         'device': None,
         'inference_bs': 10000,
         'type': ModelType.AUTOREGRESSIVE,
         'ignore_hierarchy': True,
         'find_required_models': find_flat_ar_models,
         'model_subfolder': 'ar_models'
         }
    for k, v in kwargs.items():
        p[k] = v
    return p


def ssar_params(**kwargs):
    p = flat_ar_params(**kwargs)
    p.update({'type': ModelType.SCHEMA_STRUCTURED_AR,
              'ignore_hierarchy': False,
              'self_evidence_only': False,
              'layers_hierarchy': 2,
              'average_embd': True,
              'hierarchy_embedding_layer_idxs': None,
              'find_required_models': find_ssar_models,
              'model_subfolder': 'set_ar_models',
              'max_hierarchy_tuples': 10,
              'max_hierarchy_depth': 1})
    for k, v in kwargs.items():
        p[k] = v
    return p
