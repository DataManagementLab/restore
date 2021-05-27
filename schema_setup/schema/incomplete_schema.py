import logging
import os
import pickle

import numpy as np
import pandas as pd
from spn.structure.StatisticalTypes import MetaType

from schema_setup.schema.dataset import Dataset
from schema_setup.schema.schema import Table, Schema, Relationship

logger = logging.getLogger(__name__)


class IncompleteSchema(Schema):
    def __init__(self, default_separator=None):
        Schema.__init__(self, default_separator)

    def add_relationship(self, outgoing_table_name, outgoing_attribute_names, incoming_table_name,
                         incoming_attribute_names):
        incoming_attributes, incoming_table, outgoing_attributes, outgoing_table = self.find_relationship_objects(
            incoming_attribute_names, incoming_table_name, outgoing_attribute_names, outgoing_table_name)

        relationship = IncompleteRelationship(self, outgoing_attributes, incoming_attributes)
        self.relationships.append(relationship)
        outgoing_table.outgoing_relationships.append(relationship)
        incoming_table.incoming_relationships.append(relationship)

        return relationship


class IncompleteTable(Table):
    def __init__(self, complete_table, schema, directory):
        Table.__init__(self, schema, complete_table.name, complete_table._csv_columns,
                       primary_key=[c.name for c in complete_table.primary_key],
                       irrelevant_attribute_names=complete_table._irrelevant_csv_columns,
                       filename=complete_table.filename, separator=complete_table.separator,
                       sample_rate=complete_table.sample_rate, table_size=complete_table.table_size)
        # which primary keys remain after the removal in the incomplete schema
        self._remaining_pks = None
        self._remaining_tuples = None
        self._removed_pks = None
        self._incomplete_dataset = None

        self.complete_dataset = None
        self.complete_pks = None
        self.complete = True

        self.directory = directory
        self.meta_data = None

    @property
    def remaining_pks(self):
        self.load()
        return self._remaining_pks

    @remaining_pks.setter
    def remaining_pks(self, remaining_pks):
        self.load()
        self._remaining_pks = remaining_pks
        self._removed_pks = self.complete_pks.difference(self._remaining_pks)
        self.update_complete()

    @property
    def removed_pks(self):
        self.load()
        return self._removed_pks

    @removed_pks.setter
    def removed_pks(self, remove_pks):
        self.load()
        self._removed_pks = remove_pks
        self._remaining_pks = self.complete_pks.difference(self._removed_pks)
        self.update_complete()

    def update_complete(self):
        self.complete = (len(self._remaining_pks) == len(self.complete_pks))
        df_rows = self.complete_dataset.df_rows
        remaining_tuples = df_rows[df_rows[self.primary_key[0].full_name].isin(self.remaining_pks)]
        self._incomplete_dataset = Dataset(remaining_tuples, self.complete_dataset.attributes)

    @property
    def remaining_tuples(self):
        if self._remaining_tuples is not None:
            return self._remaining_tuples

        self.load()
        pk_rows = self.complete_dataset.df_rows[self.primary_key[0].full_name]
        self._remaining_tuples = self.complete_dataset.df_rows[pk_rows.isin(self._remaining_pks)]
        return self._remaining_tuples

    def load(self):
        if self.complete_dataset is None:
            with open(os.path.join(self.directory, 'meta_data.pkl'), 'rb') as handle:
                self.meta_data = pickle.load(handle)

            df_table = pd.read_hdf(os.path.join(self.directory, f'{self.name}.hdf'), key='df')

            # drop irrelevant attributes
            relevant_attributes = set(df_table.columns).intersection([a.full_name for a in self.attributes]).difference(
                self.full_irrelevant_csv_columns)
            irrelevant_attributes = set(df_table.columns).difference(relevant_attributes)

            df_table.drop(columns=irrelevant_attributes, inplace=True)

            attributes = []
            for c in df_table.columns:
                for a in self.attributes:
                    if a.full_name == str(c):
                        attributes.append(a)
                        continue
            assert len(attributes) == len(df_table.columns)
            for a in attributes:
                a.categorical_columns_dict = self.meta_data[self.name]['categorical_columns_dict'].get(a.full_name)
                a.meta_type = MetaType.REAL if a.categorical_columns_dict is None else MetaType.DISCRETE
                assert len(self.meta_data[self.name]['relevant_attributes']) == len(
                    self.meta_data[self.name]['null_values_column'])
                a_meta_data_index = self.meta_data[self.name]['relevant_attributes'].index(a.full_name)
                a.null_value = self.meta_data[self.name]['null_values_column'][a_meta_data_index]

            self.complete_dataset = Dataset(df_table, attributes)
            df_rows = self.complete_dataset.df_rows
            self.complete_pks = set(df_rows[self.primary_key[0].full_name].unique())
            self.remaining_pks = self.complete_pks
            self._incomplete_dataset = Dataset(self.complete_dataset.df_rows, attributes)

        return self.complete_dataset

    @property
    def incomplete_dataset(self):
        self.load()

        # set tuple factors to nan when they are not known
        return self._incomplete_dataset

    @property
    def incomplete_dataset_with_tfs(self):
        self.load()

        df_rows = self.complete_dataset.df_rows
        remaining_tuples = df_rows[df_rows[self.primary_key[0].full_name].isin(self.remaining_pks)]
        return Dataset(remaining_tuples, self.complete_dataset.attributes)

    def update_unknown_tfs(self):
        for r in self.incoming_relationships:
            pk_name = self.primary_key[0].full_name
            # copy to avoid changing the complete df
            self._incomplete_dataset.df_rows = self._incomplete_dataset.df_rows.copy()

            self._incomplete_dataset.df_rows.loc[~self._incomplete_dataset.df_rows[pk_name].isin(r.keep_tuple_fks),
                                                 r.tuple_factor_name] = np.nan


class IncompleteRelationship(Relationship):
    def __init__(self, schema, outgoing_attributes, incoming_attributes):
        Relationship.__init__(self, schema, outgoing_attributes, incoming_attributes)

        # Foreign keys which have tuple factors set
        self.keep_tuple_fks = None

        # Primary keys without fk
        self.pks_without_fk = None

    def update_pks_without_fk(self):
        referenced_pk_name = self.incoming_table.primary_key[0].full_name

        referencing_table = self.outgoing_table
        referencing_fk_name = self.outgoing_attributes[0].full_name

        self.keep_tuple_fks = set(self.incoming_table.incomplete_dataset.df_rows[referenced_pk_name])

        remaining_tuples = self.outgoing_table.remaining_tuples
        tuples_without_fk = remaining_tuples[remaining_tuples[referencing_fk_name].isna()]
        self.pks_without_fk = set(tuples_without_fk[referencing_table.primary_key[0].full_name])
