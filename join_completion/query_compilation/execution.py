import logging
import os
import time

import numpy as np
import pandas as pd

from schema_setup.data_preparation.utils import load_pkl, save_pkl
from schema_setup.schema.schema_utils import stable_hash

logger = logging.getLogger(__name__)


def execute_plan(plan, scenario_directory, skip_save=False):
    """
    Executes query plan if it cannot be read from disk.

    :param query_plan:
    :param scenario_directory:
    :param query_plan_identifier:
    :return:
    """
    scenario_directory = os.path.join(scenario_directory, 'queries')
    # hash query plan identifier because of file length limits
    # hashlib for stable hashes
    query_path = os.path.join(scenario_directory, stable_hash(plan.identifier) + '.hdf')
    weight_path = os.path.join(scenario_directory, stable_hash(plan.identifier) + 'w.npy')
    execution_time_path = os.path.join(scenario_directory, stable_hash(plan.identifier) + '_t')
    stats_path = os.path.join(scenario_directory, stable_hash(plan.identifier) + '_stats')

    try:
        S = pd.read_hdf(query_path, key='df')
        weights = np.load(weight_path)
        execution_time = load_pkl(execution_time_path)
        stats = load_pkl(stats_path)

        logger.info(f"Loaded query result from {query_path}")

    except (FileNotFoundError, ValueError) as e:
        os.makedirs(scenario_directory, exist_ok=True)

        start_t = time.perf_counter()
        S, stats = plan.execute()
        execution_time = time.perf_counter() - start_t
        save_pkl(execution_time_path, execution_time)
        save_pkl(stats_path, stats)

        weights = S.weights
        if not skip_save:
            np.save(weight_path, weights)

        S = S.df_rows
        if not skip_save:
            S.to_hdf(query_path, key='df', format='table')

        logger.info(f"Saved query result to {query_path}")

    return S, weights, execution_time, stats
