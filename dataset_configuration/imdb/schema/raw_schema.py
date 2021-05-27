from schema_setup.schema.schema import Schema, Table


def gen_raw_imdb_schema():
    schema = Schema(default_separator=',')
    schema.tables.add(Table(schema, 'movie',
                            ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code',
                             'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'],
                            irrelevant_attribute_names=['imdb_index', 'imdb_id', 'phonetic_code',
                                                        'episode_of_id', 'season_nr', 'episode_nr', 'series_years',
                                                        'md5sum'],
                            filename='title.csv'
                            ))

    schema.tables.add(Table(schema, 'movie_info',
                            ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                            irrelevant_attribute_names=['note'],
                            filename='movie_info.csv'
                            ))

    schema.tables.add(Table(schema, 'movie_info_idx',
                            ['id', 'movie_id', 'info_type_id', 'info', 'note'],
                            irrelevant_attribute_names=['note'],
                            filename='movie_info_idx.csv'
                            ))

    schema.tables.add(Table(schema, 'info_type',
                            ['id', 'info'],
                            filename='info_type.csv'
                            ))

    schema.tables.add(Table(schema, 'person_info',
                            ['id', 'person_id', 'info_type_id', 'info', 'note'],
                            irrelevant_attribute_names=['note'],
                            filename='person_info.csv'
                            ))

    # no need to do anything here
    schema.tables.add(Table(schema, 'movie_companies',
                            ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                            irrelevant_attribute_names=['note'],
                            filename='movie_companies.csv'
                            ))

    schema.tables.add(Table(schema, 'company',
                            ['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                            irrelevant_attribute_names=['imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                            filename='company_name.csv'
                            ))

    schema.tables.add(Table(schema, 'cast_info',
                            ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                            irrelevant_attribute_names=['note', 'nr_order'],
                            filename='cast_info.csv'
                            ))

    schema.tables.add(Table(schema, 'name',
                            ['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf',
                             'surname_pcode', 'md5sum'],
                            irrelevant_attribute_names=['imdb_index', 'imdb_id', 'name_pcode_cf', 'name_pcode_nf',
                                                        'surname_pcode', 'md5sum'],
                            filename='name.csv'
                            ))

    return schema
