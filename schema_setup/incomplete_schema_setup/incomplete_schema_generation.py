import logging
import os
import pickle

import numpy as np

from schema_setup.incomplete_schema_setup.removal_method import RemovalMethod
from schema_setup.incomplete_schema_setup.utils import compute_correlated_vector
from schema_setup.schema.incomplete_schema import IncompleteTable, IncompleteSchema
from schema_setup.schema.schema_utils import generate_combined_scenario_name

logger = logging.getLogger(__name__)


def generate_incomplete_schema(scenario_directory, dataset, schema, projected_tables, directory, tf_removals,
                               tf_keep_rates, tuple_removal_tables, tuple_removal_keep_rates, removal_methods,
                               removal_attrs, removal_attr_values, removal_attr_biases, seed=0, cascading_deletes=False,
                               skip_save=False):
    """
    Derives an incomplete schema from a complete one

    :param skip_save:
    :param scenario_directory:
    :param dataset:
    :param schema:
    :param projected_tables:
    :param directory:
    :param tf_removals:
    :param tf_keep_rates:
    :param tuple_removal_tables:
    :param tuple_removal_keep_rates:
    :param removal_methods:
    :param removal_attrs:
    :param removal_attr_values:
    :param removal_attr_biases:
    :param seed:
    :param cascading_deletes: we will delete tuples from the tuple_removal_table. This parameter controls whether for
        referencing tables the tuples should also be deleted or the fk should only be set to NaN.
    :return:
    """
    scenario_name = generate_combined_scenario_name(dataset, projected_tables, tf_removals, tf_keep_rates,
                                                    tuple_removal_tables, tuple_removal_keep_rates, removal_methods,
                                                    removal_attrs, removal_attr_values, removal_attr_biases, seed,
                                                    cascading_deletes)
    scenario_directory = os.path.join(scenario_directory, scenario_name)
    scenario_path = os.path.join(scenario_directory, 'scenario.pkl')

    try:
        with open(scenario_path, 'rb') as handle:
            incomplete_schema = pickle.load(handle)
        logger.info(f"Loaded scenario from {scenario_path}")

    except (FileNotFoundError, EOFError, ValueError):
        logger.info(f"Could not load scenario from {scenario_path}. Creating new scenario.")

        incomplete_schema = _incomplete_schema(directory, projected_tables, schema)
        incomplete_schema = derive_incomplete_schema(cascading_deletes, removal_attrs, removal_attr_biases,
                                                     removal_attr_values, removal_methods, incomplete_schema,
                                                     tf_keep_rates, tf_removals, tuple_removal_keep_rates,
                                                     tuple_removal_tables)

        os.makedirs(scenario_directory, exist_ok=True)
        if not skip_save:
            with open(scenario_path, 'wb') as f:
                pickle.dump(incomplete_schema, f, pickle.HIGHEST_PROTOCOL)

    # report number of incomplete/complete sets and relations
    for t in incomplete_schema.tables:
        logger.info(f"Table {t}"
                    f"\n\tremaining tuples: {len(t.remaining_pks)}"
                    f"\n\tremoved tuples: {len(t.removed_pks)}"
                    f"\n\tkeep rate: {len(t.remaining_pks) / (len(t.removed_pks) + len(t.remaining_pks)) * 100:.2f}%")

    for r in incomplete_schema.relationships:
        logger.info(
            f"Relationship {r.incoming_table.primary_key[0].full_name} -> {r.outgoing_table.primary_key[0].full_name}"
            f"\n\ttf_keep_fk_values: {len(r.keep_tuple_fks)}")

    return scenario_name, incomplete_schema


def derive_incomplete_schema(cascading_deletes, removal_attrs, removal_attr_biases, removal_attr_values,
                             removal_methods, incomplete_schema, tf_keep_rates, tf_removals, tuple_removal_keep_rates,
                             tuple_removal_tables):
    for removal_attr, removal_attr_bias, removal_attr_value, removal_method, tuple_removal_table, \
        tuple_removal_keep_rate in zip(removal_attrs, removal_attr_biases, removal_attr_values, removal_methods,
                                       tuple_removal_tables, tuple_removal_keep_rates):
        # project to relevant tables
        tuple_removal_table = incomplete_schema.table_dict[tuple_removal_table]
        tuple_removal_table.load()
        dataset_join = tuple_removal_table.incomplete_dataset
        # project to required attributes
        # important: this step is absolutely necessary. It screws up the performance otherwise
        projection_attributes = [a for a in dataset_join.attributes if a.is_pk or a.full_name == removal_attr]
        dataset_join = dataset_join.project(projection_attributes)
        pk_name = tuple_removal_table.primary_key[0].full_name

        if removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:
            try:
                potential_attrs = [a for a in tuple_removal_table.attributes if a.full_name == removal_attr]
                if len(potential_attrs) == 1:
                    for name, id in potential_attrs[0].categorical_columns_dict.items():
                        if id == int(removal_attr_value):
                            logger.info(f"Biased removal for attribute value {name}")
            except:
                pass

        remaining_pk_values = compute_remove_pks(dataset_join, pk_name, tuple_removal_keep_rate, removal_method,
                                                 removal_attr, removal_attr_values, removal_attr_bias)
        # apply remaining_pk_values and propagate changes (of the removed pk's)
        pk_attribute = dataset_join.attribute_dict[pk_name]
        pk_attribute.table.remaining_pks = remaining_pk_values
        propagate_pk_removals(pk_attribute.table, cascading_deletes)

    # update for every relationships, which pks have complete sets
    for r in incomplete_schema.relationships:
        r.update_pks_without_fk()
    # remove random tfs
    for tf_name, keep_rate in zip(tf_removals, tf_keep_rates):
        logger.info(f"Biased tf removal {tf_name} {keep_rate:.2f}")
        # find tf attribute
        tf_tables = [t for t in incomplete_schema.tables if t.attribute_dict.get(tf_name) is not None]
        tf_table = tf_tables[0]

        # first sample remove fks from NaN fields otherwise it is impossible to learn the relationship of tf values and
        # the attribute values
        if len(set(tf_table.incomplete_dataset.df_rows.columns).intersection(removal_attrs)) == 0:
            fk_ids = np.unique(tf_table.incomplete_dataset.df_rows[tf_table.primary_key[0].full_name].values)
            # sample keep tfs
            tf_keep_fk_values = set(generate_remove_ids(1 - keep_rate, fk_ids, RemovalMethod.UNIFORM, None, 0.0))

        else:
            removal_attr_not_nan = list(set(tf_table.incomplete_dataset.df_rows.columns).intersection(removal_attrs))
            assert len(removal_attr_not_nan) == 1
            removal_attr_not_nan = removal_attr_not_nan[0]

            nan_fk_idx = tf_table.incomplete_dataset.df_rows[removal_attr_not_nan].isna()

            fk_ids = set(tf_table.incomplete_dataset.df_rows[tf_table.primary_key[0].full_name].values)
            no_remove_tuples = len(fk_ids) * (1 - keep_rate)
            nan_fk_ids = np.unique(
                tf_table.incomplete_dataset.df_rows.loc[nan_fk_idx, tf_table.primary_key[0].full_name].values)
            nnan_fk_ids = np.array(list(fk_ids.difference(nan_fk_ids)))

            tf_remove_values = set()
            if len(nan_fk_ids) > 0:
                keep_rate = max(1 - no_remove_tuples / len(nan_fk_ids), 0.0)
                tf_remove_values = set(generate_remove_ids(keep_rate, nan_fk_ids, RemovalMethod.UNIFORM, None, 0.0))
            if len(tf_remove_values) < no_remove_tuples and len(nnan_fk_ids) > 0:
                keep_rate = 1 - (no_remove_tuples - len(tf_remove_values)) / len(nnan_fk_ids)
                tf_remove_values.update(generate_remove_ids(keep_rate, nnan_fk_ids, RemovalMethod.UNIFORM, None, 0.0))
            tf_keep_fk_values = fk_ids.difference(tf_remove_values)
            assert np.isclose(len(tf_keep_fk_values), len(fk_ids) - no_remove_tuples, rtol=0.05)

        # find relationship
        rels = [r for r in incomplete_schema.relationships if
                r.tf_attribute == tf_table.attribute_dict.get(tf_name)]
        rels[0].keep_tuple_fks = tf_keep_fk_values
    for t in incomplete_schema.tables:
        t.update_unknown_tfs()
    return incomplete_schema


def _incomplete_schema(directory, projected_tables, schema, return_mapping=False):
    incomplete_schema = IncompleteSchema(default_separator=schema.default_separator)

    incomplete_table_mapping = {table: IncompleteTable(table, incomplete_schema, directory) for table in
                                schema.tables if
                                table.name in projected_tables}
    incomplete_schema.tables = set(incomplete_table_mapping.values())

    r_mapping = dict()
    for r in schema.relationships:
        if r.incoming_table.name in projected_tables and r.outgoing_table.name in projected_tables:
            r_mapping[r] = incomplete_schema.add_relationship(r.outgoing_table.name,
                                                              [a.name for a in r.outgoing_attributes],
                                                              r.incoming_table.name,
                                                              [a.name for a in r.incoming_attributes])

    if return_mapping:
        a_mapping = dict()
        for t in schema.tables:
            if t.name in projected_tables:
                for a, a_mapped in zip(t.attributes, incomplete_table_mapping[t].attributes):
                    assert a != a_mapped
                    a_mapping[a] = a_mapped

        return incomplete_schema, incomplete_table_mapping, r_mapping, a_mapping

    return incomplete_schema


def propagate_pk_removals(incomplete_table, cascading_deletes):
    """
    For every removed pk, check which tuples reference this pk -> these are removed pks of the other table
    (do this recursively)
    :param incomplete_table:
    :param cascading_deletes:
    :return:
    """
    for r in incomplete_table.incoming_relationships:
        # make sure we also remove pks of other tables referencing these pks
        referencing_table = r.outgoing_table
        referencing_attribute = r.outgoing_attributes[0]
        primary_key = referencing_attribute.table.primary_key[0]

        # load the dataset first
        dataset_referencing_table = referencing_table.load()
        if cascading_deletes:
            # which rows reference now removed values
            df_ref_rows = dataset_referencing_table.df_rows[
                dataset_referencing_table.df_rows[referencing_attribute.full_name].isin(incomplete_table.removed_pks)]
            # corresponding pks should be removed
            referencing_table.removed_pks = set(df_ref_rows[primary_key.full_name].unique())
        else:
            # set all fks to Nan in incomplete dataset
            referencing_table.incomplete_dataset.df_rows.loc[
                referencing_table.incomplete_dataset.df_rows[referencing_attribute.full_name].isin(
                    incomplete_table.removed_pks),
                referencing_attribute.full_name
            ] = np.nan

        # recursive check in schema
        if cascading_deletes:
            propagate_pk_removals(referencing_table, cascading_deletes)


def compute_remove_pks(dataset_join, pk_name, tuple_removal_keep_rate, removal_method, removal_attr,
                       removal_attr_values, removal_attr_bias):
    df_join = dataset_join.df_rows

    # replace nan values
    for i, a in enumerate(dataset_join.attributes):
        df_join.loc[df_join[a.full_name] == a.null_value, a.full_name] = np.nan

    # delete a random portion of the neighborhood listings for the remaining ids
    pk_candidates = set(df_join[pk_name].values)

    pk_weights = None
    if removal_method == RemovalMethod.UNIFORM:
        pk_candidates = np.array(list(pk_candidates))
    elif removal_method == RemovalMethod.BIAS or removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:

        df_pk_candidates = df_join[df_join[pk_name].isin(pk_candidates)]
        if removal_method == RemovalMethod.BIAS:
            df_pk_candidates = df_pk_candidates[[pk_name, removal_attr]].groupby([pk_name]).agg(
                {removal_attr: [np.nanmean]}).reset_index()

            pk_weights = df_pk_candidates[removal_attr].values
            pk_candidates = df_pk_candidates[pk_name].values
        elif removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:
            pk_candidates, pk_weights = categorical_removal_weights(df_pk_candidates, removal_attr, removal_attr_values,
                                                                    pk_name)

    remove_pk_values = generate_remove_ids(tuple_removal_keep_rate, pk_candidates, removal_method, pk_weights,
                                           removal_attr_bias)
    remaining_pk_values = set(pk_candidates).difference(remove_pk_values)
    logger.info(
        f"Removal"
        f"\n\tremaining {pk_name} values: {len(remaining_pk_values)}"
        f"\n\t{pk_name} keep rate: {len(remaining_pk_values) / len(pk_candidates) * 100:.2f}%")

    return remaining_pk_values


def categorical_removal_weights(df_join, removal_attr, removal_attr_values, set_pk):
    removal_attr_values = set([float(a) if a != 'none' else 'none' for a in removal_attr_values])
    nna_counts = df_join.loc[~df_join[removal_attr].isna(), [set_pk, removal_attr]] \
        .groupby([set_pk]).count()
    rel_attr_counts = df_join.loc[df_join[removal_attr].isin(removal_attr_values), [set_pk, removal_attr]] \
        .groupby([set_pk]).count()

    rel_attr_counts = (rel_attr_counts / nna_counts).reset_index()
    # happens if rel_attr_counts is 0, so no occurrence
    rel_attr_counts[removal_attr] = rel_attr_counts[removal_attr].fillna(value=0)
    fk_ids = rel_attr_counts[set_pk].values
    fk_weights = rel_attr_counts[removal_attr].values

    # add ids which are all nan (i.e., did not appear in nna_counts)
    remaining_ids = np.array(list(set(df_join[set_pk]).difference(fk_ids)))
    fk_ids = np.concatenate((fk_ids, remaining_ids))
    fk_weights = np.concatenate((fk_weights, np.zeros(len(remaining_ids))))
    if len(remaining_ids) > 0:
        fk_weights[-len(remaining_ids):] = np.nan

    return fk_ids, fk_weights


def generate_remove_ids(keep_rate, id_values, removal_method, weights, correlation):
    if correlation is not None and correlation < 0:
        keep_ids = generate_remove_ids(1 - keep_rate, id_values, removal_method, weights, -correlation)
        return set(id_values).difference(keep_ids)

    if removal_method == RemovalMethod.UNIFORM:
        no_complete_id_values = int(keep_rate * len(id_values))
        keep_idx = np.random.choice(len(id_values), size=no_complete_id_values, replace=False)
        incomplete_id_values = set(np.delete(id_values, keep_idx))

    elif removal_method == RemovalMethod.BIAS or removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:
        assert len(weights) == len(id_values)
        assert correlation <= 1.0

        weights = weights.reshape(-1)
        idx_not_nan = ~np.isnan(weights)
        no_not_nan = len(np.where(idx_not_nan)[0])
        weights = weights[idx_not_nan]

        # probability of getting selected is correlated with weights
        p = compute_correlated_vector(weights, correlation, probability_distribution=True, normalize_input=True)

        assert len(p[p >= 0]) >= int(keep_rate * no_not_nan), "Fallback strategy for this case not yet implemented"
        keep_idx_not_nan = np.random.choice(no_not_nan, size=int(keep_rate * no_not_nan), replace=False, p=p)
        incomplete_id_values = set(id_values)
        incomplete_id_values = incomplete_id_values.difference(id_values[idx_not_nan][keep_idx_not_nan])

        # random attributes with null values
        keep_idx_nan = np.random.choice(len(id_values) - no_not_nan,
                                        size=int(keep_rate * (len(id_values) - no_not_nan)), replace=False)
        incomplete_id_values = incomplete_id_values.difference(id_values[~idx_not_nan][keep_idx_nan])
    else:
        raise NotImplementedError(f"removal_method {removal_method} not implemented")

    return incomplete_id_values
