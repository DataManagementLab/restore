from schema_setup.schema.schema import Attribute

import numpy as np
import pandas as pd


class Dataset:
    """
    Rows of a table or join. Used for learning.
    """

    def __init__(self, df_rows, attributes, weights=None):
        self.df_rows = df_rows
        self.attributes = [a for a in attributes]
        self.attribute_dict = {a.full_name: a for a in self.attributes}
        if weights is None:
            weights = np.ones(len(df_rows))
        self.weights = weights
        assert len(df_rows.columns) == len(attributes)
        assert len(self.weights) == len(df_rows)

    def join(self, right_dataset, relationship, how='inner'):
        self.df_rows = self.df_rows.reset_index(drop=True).reset_index()
        if relationship.incoming_attributes[0] in self.attributes:
            df_joined = self.df_rows.merge(right_dataset.df_rows, left_on=relationship.incoming_attributes[0].full_name,
                                           right_on=relationship.outgoing_attributes[0].full_name, how=how)
        else:
            df_joined = self.df_rows.merge(right_dataset.df_rows, left_on=relationship.outgoing_attributes[0].full_name,
                                           right_on=relationship.incoming_attributes[0].full_name, how=how)
        weights = self.weights[df_joined['index']]
        df_joined.drop(columns=['index'], inplace=True)
        self.df_rows.drop(columns=['index'], inplace=True)

        return Dataset(df_joined, self.attributes + right_dataset.attributes, weights=weights)

    def project(self, projection_attributes):
        assert set(self.attributes).issuperset(projection_attributes), "Not all projection attributes are available"
        proj_attributes = [a for a in self.attributes if a in projection_attributes]
        proj_rows = self.df_rows[[a.full_name for a in projection_attributes]]

        return Dataset(proj_rows, proj_attributes, weights=self.weights)

    def concat(self, other_dataset):
        k = {}
        for a in self.attributes:
            k[a.full_name] = a
        for a in other_dataset.attributes:
            k[a.full_name] = a

        concat_df_rows = pd.concat((self.df_rows, other_dataset.df_rows))
        merged_attributes = [k[str(col)] for col in concat_df_rows.columns]

        return Dataset(concat_df_rows, merged_attributes, weights=np.concatenate((self.weights, other_dataset.weights)))

    def augment_current_tuple_factors(self, next_table, r):
        if 'current_tf' in self.df_rows.columns:
            self.df_rows.drop(columns='current_tf', inplace=True)
            self.attributes.remove(self.attribute_dict['current_tf'])

        self.df_rows = self.df_rows.reset_index(drop=True).reset_index()
        muls = self.df_rows.merge(next_table.df_rows, left_on=r.incoming_attributes[0].full_name,
                                  right_on=r.outgoing_attributes[0].full_name, how='inner')
        muls.set_index(['index'], drop=False, inplace=True)
        muls = muls[['index']]
        muls.columns = ['current_tf']
        muls = muls.groupby(muls.index).count()
        self.df_rows = self.df_rows.merge(muls, left_index=True, right_index=True, how='left')
        self.df_rows['current_tf'] = self.df_rows['current_tf'].fillna(0)
        self.df_rows.drop(columns=['index'], inplace=True)
        current_tf_attribute = Attribute(None, 'current_tf', is_current_tf=True)
        self.attributes.append(current_tf_attribute)
        self.attribute_dict['current_tf'] = current_tf_attribute

        assert len(self.attributes) == len(self.df_rows.columns)
        return self

    @property
    def tables(self):
        return {a.table for a in self.attributes}
