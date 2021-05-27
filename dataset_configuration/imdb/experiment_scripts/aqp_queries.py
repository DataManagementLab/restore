from evaluation.approximate_query_processing.aqp_queries import AQPQuery, Operator

imdb_aqp_queries = {
    'movie.country': {2: AQPQuery(where_conditions=[('movie.genre', Operator.EQ, 4)],
                                  grouping_attributes=['movie.country'],
                                  completion_tables=['movie']),
                      4: AQPQuery(where_conditions=[('company.country_code', Operator.EQ, 1)],
                                  grouping_attributes=['movie.production_year'],
                                  completion_tables=['movie', 'movie_companies', 'company']),
                      },
    'movie.genre': {0: AQPQuery(where_conditions=[('movie.genre', Operator.EQ, 4)],
                                grouping_attributes=['movie.production_year'],
                                completion_tables=['movie']),
                    4: AQPQuery(where_conditions=[('company.country_code', Operator.EQ, 1)],
                                grouping_attributes=['movie.production_year'],
                                completion_tables=['movie', 'movie_companies', 'company']),
                    5: AQPQuery(where_conditions=[],
                                grouping_attributes=['company.country_code'],
                                completion_tables=['movie', 'movie_companies', 'company']),
                    },
    'movie.production_year': {1: AQPQuery(where_conditions=[],
                                          grouping_attributes=['movie.production_year'],
                                          completion_tables=['movie']),
                              4: AQPQuery(aggregation_attribute='movie.production_year',
                                          where_conditions=[('director.birth_country', Operator.EQ, 7)],
                                          grouping_attributes=['movie.production_year'],
                                          completion_tables=['movie', 'movie_director', 'director']),
                              },
    'director.birth_year': {0: AQPQuery(aggregation_attribute='director.birth_year',
                                        where_conditions=[('director.gender', Operator.EQ, 1)],
                                        grouping_attributes=[],
                                        completion_tables=['director']),
                            4: AQPQuery(where_conditions=[('director.gender', Operator.EQ, 1)],
                                        grouping_attributes=[],
                                        completion_tables=['movie', 'movie_director', 'director']),
                            },
    'company.country_code': {1: AQPQuery(where_conditions=[('company.country_code', Operator.EQ, 1)],
                                         grouping_attributes=[],
                                         completion_tables=['company']),
                             }
}