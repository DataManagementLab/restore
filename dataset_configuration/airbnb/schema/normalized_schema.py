from schema_setup.schema.schema import Schema, Table


def gen_normalized_airbnb_schema():
    schema = Schema(default_separator=';')
    schema.tables.add(Table(schema, 'listings',
                            ['id', 'host_id', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
                             'beds', 'bed_type', 'square_feet', 'price', 'weekly_price', 'monthly_price',
                             'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
                             'maximum_nights', 'has_availability', 'availability_30', 'availability_60',
                             'availability_90', 'availability_365', 'cancellation_policy', 'reviews_per_month',
                             'neighborhood_id'],
                            irrelevant_attribute_names=['weekly_price', 'monthly_price',
                                                        'has_availability', 'availability_30', 'availability_60',
                                                        'availability_90', 'maximum_nights', 'square_feet',
                                                        'reviews_per_month', 'bathrooms', 'bedrooms',
                                                        'beds', 'bed_type'],
                            primary_key=['id'], filename='listings.csv', table_size=494928.0))

    schema.tables.add(Table(schema, 'neighborhoods',
                        ['neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'state', 'country',
                         'neighborhood_id'], filename='neighborhoods.csv',
                        primary_key=['neighborhood_id'], table_size=8005.0))

    schema.tables.add(Table(schema, 'hosts',
                            ['host_id', 'host_since', 'host_location', 'host_response_time', 'host_response_rate',
                             'host_acceptance_rate', 'host_neighbourhood'], filename='hosts.csv',
                            primary_key=['host_id'], table_size=363133.0))

    schema.add_relationship('listings', ['host_id'], 'hosts', ['host_id'])
    schema.add_relationship('listings', ['neighborhood_id'], 'neighborhoods', ['neighborhood_id'])

    return schema
