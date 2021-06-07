import logging
import os
import pickle
import shutil
from time import perf_counter

import numpy as np

from schema_setup.data_preparation.utils import read_table_csv
from schema_setup.incomplete_schema_setup.utils import common_element

logger = logging.getLogger(__name__)


def prepare_single_table(table, normalized_data_directory, hdf_data_directory, max_distinct_vals=10000,
                         max_table_data=20000000, replace_nans=True):
    """
    Reads table csv. Adds multiplier fields, missing value imputation, dict for categorical data. Adds null tuple tables.

    :param table: table object in schema for which csv is read
    :param normalized_data_directory: directory where csv files are stored
    :param hdf_data_directory: directory where hdf files should be written to
    :param max_distinct_vals: maximum number of distinct values until a column is considered irrelevant
    :param max_table_data: maximum number of tuples which are written to the hdf file

    :return:
    """
    table_meta_data = dict()

    table_data = read_table_csv(os.path.join(normalized_data_directory, table.filename), table.full_csv_columns,
                                table.full_irrelevant_csv_columns, table.separator)
    logger.info(f"\t table size: {len(table_data)}")

    hdf_path = os.path.join(hdf_data_directory, table.name + '.hdf')
    table_meta_data['hdf_path'] = hdf_path
    table_meta_data['incoming_relationship_means'] = {}

    # we can ignore functional dependencies

    # add multiplier fields
    tuple_factor_columns = set()
    logger.info("Preparing multipliers for table {}".format(table.name))
    for relationship in table.incoming_relationships:
        logger.info("Preparing multiplier {} for table {}".format(relationship, table))

        neighbor_table = relationship.outgoing_table
        assert len(relationship.incoming_attributes) == 1 and \
               len(table.primary_key) == 1 and \
               table.primary_key[0] == relationship.incoming_attributes[0], \
            "Currently, only single primary keys are supported for table with incoming edges"

        left_attribute = relationship.incoming_attributes[0]
        right_attribute = relationship.outgoing_attributes[0]
        primary_key = table.primary_key[0]

        neighbor_table_data = read_table_csv(os.path.join(normalized_data_directory, neighbor_table.filename),
                                             neighbor_table.full_csv_columns,
                                             neighbor_table.full_irrelevant_csv_columns,
                                             neighbor_table.separator).set_index(right_attribute.full_name, drop=False)

        table_data = table_data.set_index(left_attribute.full_name, drop=False)

        # fix for new pandas version
        table_data.index.name = None
        neighbor_table_data.index.name = None
        muls = table_data.join(neighbor_table_data, how='left')[[primary_key.full_name, right_attribute.full_name]] \
            .groupby([primary_key.full_name]).count()

        tuple_factor_col_name = relationship.tuple_factor_name
        tuple_factor_columns.add(tuple_factor_col_name)

        muls.columns = [tuple_factor_col_name]
        # if we just have a sample of the neighbor table we assume larger multipliers
        muls[tuple_factor_col_name] = muls[tuple_factor_col_name] * 1 / neighbor_table.sample_rate

        table_data = table_data.join(muls)
        # find tuple factor percentiles
        for i in range(1, 10):
            percentage = i * 10
            logger.info(
                f"\ttuple factor percentile ({percentage}%): "
                f"{np.percentile(table_data[tuple_factor_col_name], percentage):.2f}")
        logger.info(
            f"\ttuple factor percentile ({99}%): "
            f"{np.percentile(table_data[tuple_factor_col_name], 99):.2f}")
        logger.info(f"\ttuple factor percentile ({100}%): {np.max(table_data[tuple_factor_col_name]):.2f}")
        table_meta_data['incoming_relationship_means'][relationship.identifier] = table_data[
            tuple_factor_col_name].mean()

    # save if there are entities without FK reference (e.g. orders without customers)
    for relationship in table.outgoing_relationships:
        fk_attribute_name = relationship.outgoing_attributes[0].full_name

        table_meta_data[relationship.tuple_factor_name] = {
            'fk_attribute_name': fk_attribute_name,
            'length': table_data[fk_attribute_name].isna().sum() * 1 / table.sample_rate,
            'path': None
        }

    # null value imputation and categorical value replacement
    logger.info("Preparing categorical values and null values for table {}".format(table))
    table_meta_data['categorical_columns_dict'] = {}
    table_meta_data['null_values_column'] = []

    del_cat_attributes = []

    for attribute in table.attributes:

        rel_attribute = attribute.full_name

        # categorical value
        if table_data.dtypes[rel_attribute] == object:
            # assert attribute not in table.id_attributes, "Id columns should not contain strings"

            logger.info("\t\tPreparing categorical values for column {}".format(rel_attribute))

            if len(table_data[rel_attribute].unique()) > max_distinct_vals:
                del_cat_attributes.append(rel_attribute)
                logger.info("Ignoring column {} for table {} because "
                            "there are too many categorical values".format(rel_attribute, table))
            # all values nan does not provide any information
            elif not table_data[rel_attribute].notna().any():
                del_cat_attributes.append(rel_attribute)
                logger.info("Ignoring column {} for table {} because all values are nan".format(rel_attribute, table))
            else:
                # if replace_ground_truth_nans:
                #    table_data[rel_attribute] = CategoricalImputer().fit_transform(table_data[rel_attribute])
                #    assert table_data[rel_attribute].notna().all()
                distinct_vals = table_data[rel_attribute].unique()

                val_dict = dict(zip(distinct_vals, range(1, len(distinct_vals) + 1)))
                val_dict[np.nan] = np.nan
                table_meta_data['categorical_columns_dict'][rel_attribute] = val_dict
                nan_before = table_data[rel_attribute].isna().any()
                table_data[rel_attribute] = table_data[rel_attribute].map(val_dict.get)
                assert table_data[rel_attribute].isna().any() == nan_before

                if replace_nans:
                    table_data[rel_attribute] = table_data[rel_attribute].fillna(0)

                val_dict[np.nan] = 0
                # apparently slow
                # table_data[attribute] = table_data[attribute].replace(val_dict)
                table_meta_data['null_values_column'].append(val_dict[np.nan])

        # numerical value
        else:
            logger.info("\t\tPreparing numerical values for column {}".format(rel_attribute))

            # all nan values
            if not table_data[rel_attribute].notna().any():
                del_cat_attributes.append(rel_attribute)
                logger.info("Ignoring column {} for table {} because all values are nan".format(rel_attribute, table))
            else:
                # if replace_ground_truth_nans:
                #    table_data[rel_attribute] = table_data[rel_attribute].fillna(table_data[rel_attribute].mean())
                #    assert table_data[rel_attribute].notna().all()

                contains_nan = table_data[rel_attribute].isna().any()

                # not the best solution but works
                unique_null_val = table_data[rel_attribute].mean() + 0.0001
                assert not (table_data[rel_attribute] == unique_null_val).any()

                if replace_nans:
                    table_data[rel_attribute] = table_data[rel_attribute].fillna(unique_null_val)
                    if contains_nan:
                        assert (table_data[rel_attribute] == unique_null_val).any(), "Null value cannot be found"
                table_meta_data['null_values_column'].append(unique_null_val)

        try:
            logger.info(f"Most common element: {common_element(table_data[rel_attribute])}")
            logger.info(f"Least common element: {common_element(table_data[rel_attribute], least_common=True)}")
        except TypeError:
            pass
        # logger.info(f"element_1: {np.where(table_data[rel_attribute] == 1)[0]}")

    # remove categorical columns with too many entries from relevant tables and dataframe
    relevant_attributes = [x.full_name for x in table.attributes if x.full_name not in del_cat_attributes]
    logger.info("Relevant attributes for table {} are {}".format(table, relevant_attributes))
    logger.info("NULL values for table {} are {}".format(table, table_meta_data['null_values_column']))
    # del_cat_attributes = [table + '.' + rel_attribute for rel_attribute in del_cat_attributes]
    table_data = table_data.drop(columns=del_cat_attributes)

    assert len(relevant_attributes) == len(table_meta_data['null_values_column']), \
        "Length of NULL values does not match"
    table_meta_data['relevant_attributes'] = relevant_attributes
    table_meta_data['length'] = len(table_data) * 1 / table.sample_rate

    if replace_nans:
        assert not table_data.isna().any().any(), "Still contains null values"

    # save modified table
    if len(table_data) < max_table_data:
        table_data.to_hdf(hdf_path, key='df', format='table')
    else:
        table_data.sample(max_table_data).to_hdf(hdf_path, key='df', format='table')

    # add table parts without join partners
    logger.info("Adding table parts without join partners for table {}".format(table))
    for relationship in table.incoming_relationships:

        logger.info("Adding table parts without join partners "
                    "for table {} and relationship {}".format(table, relationship))

        neighbor_table = relationship.outgoing_table
        neighbor_primary_key = neighbor_table.primary_key[0]

        left_attribute = relationship.incoming_attributes[0]
        right_attribute = relationship.outgoing_attributes[0]

        table_data = table_data.set_index(left_attribute.full_name, drop=False)
        neighbor_table_data = read_table_csv(os.path.join(normalized_data_directory, neighbor_table.filename),
                                             neighbor_table.full_csv_columns,
                                             neighbor_table.full_irrelevant_csv_columns,
                                             neighbor_table.separator).set_index(right_attribute.full_name, drop=False)
        null_tuples = table_data.join(neighbor_table_data, how='left')
        null_tuples = null_tuples.loc[null_tuples[neighbor_primary_key.full_name].isna(), relevant_attributes]
        if len(null_tuples) > 0 and neighbor_table.sample_rate < 1:
            logger.warning(
                f"For {relationship.identifier} {len(null_tuples)} tuples without a join partner were "
                f"found. This is potentially due to the sampling rate of {neighbor_table.sample_rate}.")

        if len(null_tuples) > 0:
            null_tuple_path = os.path.join(hdf_data_directory, relationship.identifier + '.hdf')
            table_meta_data[relationship.identifier] = {
                'length': len(null_tuples) * 1 / table.sample_rate,
                'path': null_tuple_path
            }
            null_tuples.to_hdf(null_tuple_path, key='df', format='table')

    return table_meta_data


def prepare_all_tables(schema, normalized_data_directory, hdf_data_directory, max_table_data=20000000,
                       replace_nans=True, force=False):
    if not os.path.exists(os.path.join(hdf_data_directory, 'build_time_hdf.txt')) or force:
        # empty directory
        logger.info(f"Generating HDF files for tables in {normalized_data_directory} and store to "
                    f"path {hdf_data_directory}")

        if os.path.exists(hdf_data_directory):
            logger.info(f"Removing target path {hdf_data_directory}")
            shutil.rmtree(hdf_data_directory)

        logger.info(f"Making target path {hdf_data_directory}")
        os.makedirs(hdf_data_directory)

        # actual table preparation
        prep_start_t = perf_counter()
        meta_data = {}
        for table in schema.tables:
            logger.info("Preparing hdf file for table {}".format(table.name))
            meta_data[table.name] = prepare_single_table(table, normalized_data_directory,
                                                         hdf_data_directory, max_table_data=max_table_data,
                                                         replace_nans=replace_nans)

        with open(os.path.join(hdf_data_directory, 'meta_data.pkl'), 'wb') as f:
            pickle.dump(meta_data, f, pickle.HIGHEST_PROTOCOL)
        prep_end_t = perf_counter()

        with open(os.path.join(hdf_data_directory, 'build_time_hdf.txt'), 'w') as text_file:
            text_file.write(str(round(prep_end_t - prep_start_t)))

        logger.info(f"Files successfully created")

        return meta_data
