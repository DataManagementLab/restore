import os

import numpy as np
from pandas import DataFrame

from schema_setup.incomplete_schema_setup.utils import zipf, common_element


def compute_skewed_correlated(skew, correlation, discrete_steps, no_tuples, tf_constant, is_set_correlation=False):
    complete = np.empty((no_tuples, 2), dtype=np.float)
    complete[:, 0] = np.arange(no_tuples)

    attribute_a_values = zipf(abs(skew), 1, discrete_steps + 1, no_tuples)

    complete[:, 1] = attribute_a_values
    # generate correlated attribute b values
    incomplete = np.empty((no_tuples * tf_constant, 3), dtype=np.float)
    # id
    incomplete[:, 0] = np.arange(no_tuples * tf_constant)
    for partner_number in range(tf_constant):
        start_idx = partner_number * no_tuples
        end_idx = (partner_number + 1) * no_tuples

        # complete_id
        incomplete[start_idx:end_idx, 2] = np.arange(no_tuples)
        incomplete[start_idx:end_idx, 1] = attribute_a_values

        # add non-correlated random values
        if correlation < 1.0:
            no_random = int(no_tuples * (1 - correlation))
            random_idx = np.random.choice(np.arange(start_idx, end_idx), no_random, replace=False)
            # incomplete[random_idx, 1] = np.random.choice(discrete_steps, no_random)
            incomplete[random_idx, 1] = zipf(abs(skew), 1, discrete_steps + 1, no_random)

    # remove all correlation present in the data, it should be a set correlation
    if is_set_correlation:
        complete[:, 1] = zipf(abs(skew), 1, discrete_steps + 1, no_tuples)

    incomplete[0, 1] = common_element(incomplete[:, 1])
    incomplete_df = DataFrame(incomplete)
    incomplete_df.iloc[:, 1] = np.char.add('v', incomplete[:, 1].astype(str))
    incomplete = incomplete_df

    return complete, incomplete


def generate_synthetic_data(raw_dir, output_dir, no_tuples=10000, tf_constant=5, skew=1.0, correlation=0.0,
                            fanout_correlation=0.0, discrete_steps=1000, seed=0):
    np.random.seed(seed)

    is_set_correlation = False
    if fanout_correlation > 0:
        assert correlation == 0
        is_set_correlation = True
        correlation = fanout_correlation
    complete, incomplete = compute_skewed_correlated(skew, correlation, discrete_steps, no_tuples, tf_constant,
                                                     is_set_correlation=is_set_correlation)
    # save as csv
    os.makedirs(output_dir, exist_ok=True)
    DataFrame(complete).to_csv(os.path.join(output_dir, 'complete.csv'), index=False, header=True, sep=';')
    DataFrame(incomplete).to_csv(os.path.join(output_dir, 'incomplete.csv'), index=False, header=True, sep=';')
