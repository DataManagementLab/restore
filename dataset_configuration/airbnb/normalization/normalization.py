import os

import numpy as np

from dataset_configuration.airbnb.schema.unnormalized_schema import gen_unnormalized_airbnb_schema
from schema_setup.data_preparation.utils import read_table_csv


def normalize_airbnb_schema(airbnb_raw_dir, airbnb_output_dir, drop_zip=True):
    os.makedirs(airbnb_output_dir, exist_ok=True)
    schema = gen_unnormalized_airbnb_schema()

    # normalize listings table
    listings_table = schema.table_dict['listings']
    listing_data = read_table_csv(os.path.join(airbnb_raw_dir, listings_table.filename),
                                  listings_table.full_csv_columns,
                                  listings_table.full_irrelevant_csv_columns, listings_table.separator,
                                  ensure_numeric=['listings.id', 'listings.host_id'])

    host_data, listing_data = normalize_hosts(listing_data)
    listing_data, neighborhoods = normalize_neighborhoods(listing_data, drop_zip)

    host_data.to_csv(os.path.join(airbnb_output_dir, 'hosts.csv'), header=False, index=False, sep=';')
    neighborhoods.to_csv(os.path.join(airbnb_output_dir, 'neighborhoods.csv'), header=False, index=False, sep=';')
    listing_data.to_csv(os.path.join(airbnb_output_dir, 'listings.csv'), header=False, index=False, sep=';')


def normalize_neighborhoods(listing_data, drop_zip):
    neighbourhood_attributes = ['listings.neighbourhood_cleansed', 'listings.neighbourhood_group_cleansed',
                                'listings.state', 'listings.zipcode', 'listings.country']
    if drop_zip:
        neighbourhood_attributes.remove('listings.zipcode')
        listing_data.drop('listings.zipcode', axis=1, inplace=True)

    neighborhood_data = listing_data[neighbourhood_attributes]
    identified_neighborhoods = dict()
    neighborhoods_mapping = np.zeros(len(listing_data), dtype=int)
    for i, row in enumerate(neighborhood_data.itertuples(index=False)):
        if row not in identified_neighborhoods.keys():
            identified_neighborhoods[row] = len(identified_neighborhoods)
        neighborhoods_mapping[i] = identified_neighborhoods[row]
    listing_data['listings.neighborhood_id'] = neighborhoods_mapping
    neighborhoods = listing_data[neighbourhood_attributes + ['listings.neighborhood_id']].drop_duplicates()
    assert len(neighborhoods) == len(identified_neighborhoods)
    listing_data.drop(neighbourhood_attributes, axis=1, inplace=True)
    return listing_data, neighborhoods


def normalize_hosts(listing_data):
    host_attributes = [attribute for attribute in listing_data.columns if attribute.startswith('listings.host')]
    host_data = listing_data[host_attributes].drop_duplicates()
    for attribute in host_attributes:
        if attribute == 'listings.host_id':
            continue
        listing_data.drop(attribute, axis=1, inplace=True)

    # replace host_since by year
    host_data['listings.host_since'] = host_data['listings.host_since'].str.split('-', expand=True)[0]
    return host_data, listing_data
