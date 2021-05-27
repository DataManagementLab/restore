from schema_setup.schema.schema import Schema, Table


def gen_unnormalized_airbnb_schema():
    schema = Schema(default_separator=';')
    schema.tables.add(Table(schema, 'listings',
                            ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary', 'space',
                             'description', 'experiences_offered', 'neighborhood_overview', 'notes', 'transit',
                             'access', 'interaction', 'house_rules', 'thumbnail_url', 'medium_url', 'picture_url',
                             'xl_picture_url', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location',
                             'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                             'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_listings_count',
                             'host_total_listings_count', 'host_verifications', 'street', 'neighbourhood',
                             'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state', 'zipcode',
                             'market', 'smart_location', 'country_code', 'country', 'latitude', 'longitude',
                             'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
                             'amenities', 'square_feet', 'price', 'weekly_price', 'monthly_price', 'security_deposit',
                             'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
                             'calendar_updated', 'has_availability', 'availability_30', 'availability_60',
                             'availability_90', 'availability_365', 'calendar_last_scraped', 'number_of_reviews',
                             'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy',
                             'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                             'review_scores_location', 'review_scores_value', 'license', 'jurisdiction_names',
                             'cancellation_policy', 'calculated_host_listings_count', 'reviews_per_month',
                             'geolocation', 'features'],
                            irrelevant_attribute_names=['listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',
                                                        'space', 'description', 'experiences_offered',
                                                        'neighborhood_overview', 'notes', 'transit', 'access',
                                                        'interaction', 'house_rules', 'thumbnail_url', 'medium_url',
                                                        'picture_url', 'xl_picture_url', 'host_url', 'host_name',
                                                        'host_about', 'host_thumbnail_url', 'host_picture_url',
                                                        'host_verifications', 'street', 'neighbourhood', 'city',
                                                        'market', 'smart_location', 'country_code', 'amenities',
                                                        'calendar_updated', 'calendar_last_scraped', 'first_review',
                                                        'last_review', 'review_scores_rating', 'review_scores_accuracy',
                                                        'review_scores_cleanliness', 'review_scores_checkin',
                                                        'review_scores_communication', 'review_scores_location',
                                                        'review_scores_value', 'geolocation', 'features',
                                                        'calculated_host_listings_count', 'number_of_reviews',
                                                        'license', 'jurisdiction_names', 'latitude', 'longitude',
                                                        'host_listings_count', 'host_total_listings_count'],
                            filename='listings.csv'
                            ))

    return schema
