from schema_setup.schema.schema import Schema, Table


def gen_processed_imdb_schema():
    schema = Schema(default_separator=',')
    schema.tables.add(Table(schema, 'movie',
                            ['id', 'title', 'kind_id', 'production_year', 'runtime', 'genre', 'country',
                             'rating'],
                            irrelevant_attribute_names=[],
                            filename='movie.csv',
                            primary_key=['id'],
                            table_size=2528311.0
                            ))

    # no need to do anything here
    schema.tables.add(Table(schema, 'movie_companies',
                            ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                            irrelevant_attribute_names=['note'],
                            filename='movie_companies.csv',
                            primary_key=['id'],
                            table_size=2609129.0
                            ))

    schema.tables.add(Table(schema, 'company',
                            ['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                            irrelevant_attribute_names=['imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                            filename='company.csv',
                            primary_key=['id'],
                            table_size=234997.0
                            ))

    schema.add_relationship('movie_companies', ['movie_id'], 'movie', ['id'])
    schema.add_relationship('movie_companies', ['company_id'], 'company', ['id'])

    # for entity in ['actor', 'producer', 'writer', 'director']:
    for entity, entity_table_size, entity_rel_size in [('actor', 2695357., 20122660.), ('director', 299040., 1703543.)]:
        schema.tables.add(Table(schema, f'movie_{entity}',
                                ['id', 'person_id', 'movie_id', 'person_role_id'],
                                irrelevant_attribute_names=[],
                                filename=f'movie_{entity}.csv',
                                primary_key=['id'],
                                table_size=entity_table_size
                                ))

        schema.tables.add(Table(schema, f'{entity}',
                                ['id', 'name', 'gender', 'birth_year', 'birth_country'],
                                irrelevant_attribute_names=['name'],
                                filename=f'{entity}.csv',
                                primary_key=['id'],
                                table_size=entity_rel_size
                                ))

        schema.add_relationship(f'movie_{entity}', ['movie_id'], 'movie', ['id'])
        schema.add_relationship(f'movie_{entity}', ['person_id'], f'{entity}', ['id'])

    return schema
