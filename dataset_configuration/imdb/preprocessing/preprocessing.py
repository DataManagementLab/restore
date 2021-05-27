import os
from shutil import copyfile

from schema_setup.data_preparation.utils import read_table_csv
from dataset_configuration.imdb.schema.raw_schema import gen_raw_imdb_schema


def read_csv(raw_dir, table, ensure_numeric=None):
    if ensure_numeric is None:
        ensure_numeric = []
    return read_table_csv(os.path.join(raw_dir, table.filename),
                          table.full_csv_columns,
                          table.full_irrelevant_csv_columns, table.separator, ensure_numeric=ensure_numeric)


def preprocess_imdb_schema(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    schema = gen_raw_imdb_schema()

    augment_movies(output_dir, raw_dir, schema)
    persons = augment_persons(raw_dir, schema)
    cast_info = read_csv(raw_dir, schema.table_dict['cast_info'])

    # check role_type.csv for details
    preprocess_directors(output_dir, persons, cast_info, {1, 2}, 'actor')
    preprocess_directors(output_dir, persons, cast_info, {8}, 'director')

    copy_movie_companies(output_dir, raw_dir, schema)


def preprocess_directors(output_dir, persons, cast_info, types, entity_name):
    # focus on certain type
    cast_info = cast_info[cast_info['cast_info.role_id'].isin(types)]

    person_ids = cast_info[['cast_info.id', 'cast_info.person_id']]
    person_ids.set_index(['cast_info.person_id'])
    persons = persons.merge(person_ids, left_on='name.id', right_on='cast_info.person_id')
    persons.drop(columns=['cast_info.id', 'cast_info.person_id'], inplace=True)
    persons.drop_duplicates(subset=[f'name.id'], inplace=True)

    cast_info = cast_info.drop(columns=['cast_info.role_id'])

    persons.to_csv(os.path.join(output_dir, f'{entity_name}.csv'), header=False, index=False, sep=',')
    cast_info.to_csv(os.path.join(output_dir, f'movie_{entity_name}.csv'), header=False, index=False, sep=',')


def copy_movie_companies(output_dir, raw_dir, schema):
    movie_companies = schema.table_dict['movie_companies']
    copyfile(os.path.join(raw_dir, movie_companies.filename), os.path.join(output_dir, movie_companies.filename))
    company_name = schema.table_dict['company']
    copyfile(os.path.join(raw_dir, company_name.filename), os.path.join(output_dir, 'company.csv'))


def augment_persons(raw_dir, schema):
    # person info
    # info_type

    # enhance movie table with additional attributes
    # join with movie_info and info_type

    person_data_full = read_csv(raw_dir, schema.table_dict['person_info'],
                                ensure_numeric=['person_info.id', 'person_info.info_type_id'])
    name = read_csv(raw_dir, schema.table_dict['name'])

    name['name.person_id'] = name['name.id']
    name = name.set_index(['name.person_id'], drop=False)

    name = augment_attribute(21, 'name.birth_year', person_data_full, name, prefix='person_info',
                             id='person_id', isolate_last_digits=True)
    name = augment_attribute(20, 'name.birth_country', person_data_full, name,
                             prefix='person_info', id='person_id', seperator_cut=',', strip_brackets=True)

    name = name.drop(columns=['name.person_id'])
    return name


def augment_movies(output_dir, raw_dir, schema):
    # enhance movie table with additional attributes
    # join with movie_info and info_type
    movie_data = read_csv(raw_dir, schema.table_dict['movie'], ensure_numeric=['movie.id', 'movie.kind_id'])
    # maybe restrict to movies
    # movie_data = movie_data[movie_data['movie.kind_id'] == 1]
    movie_data = movie_data.set_index(['movie.id'], drop=False)
    movie_info_data = read_csv(raw_dir, schema.table_dict['movie_info'],
                               ensure_numeric=['movie_info.id', 'movie_info.movie_id', 'movie_info.info_type_id'])
    movie_data = augment_attribute(1, 'movie.runtime', movie_info_data, movie_data, isolate_last_digits=True)
    movie_data = augment_attribute(3, 'movie.genre', movie_info_data, movie_data)
    movie_data = augment_attribute(8, 'movie.country', movie_info_data, movie_data)
    del movie_info_data
    movie_info_idx_data = read_csv(raw_dir, schema.table_dict['movie_info_idx'])
    movie_data = augment_attribute(101, 'movie.rating', movie_info_idx_data, movie_data, prefix='movie_info_idx')
    del movie_info_idx_data
    movie_data.to_csv(os.path.join(output_dir, 'movie.csv'), header=False, index=False, sep=',')


def augment_attribute(type_id, attribute_name, movie_info_data, movie_data, prefix='movie_info', id='movie_id',
                      isolate_last_digits=False, seperator_cut=None, strip_brackets=True):
    additional_info = movie_info_data[movie_info_data[f'{prefix}.info_type_id'] == type_id]
    additional_info = additional_info[[f'{prefix}.{id}', f'{prefix}.info']]
    additional_info.columns = [f'{prefix}.{id}', attribute_name]
    additional_info.drop_duplicates(subset=[f'{prefix}.{id}'], inplace=True)
    additional_info = additional_info.set_index(f'{prefix}.{id}', drop=True)

    if seperator_cut is not None:
        temp = additional_info[attribute_name].str.split(seperator_cut)
        temp = temp.apply(lambda x: x[-1])
        if strip_brackets:
            temp = temp.str.split('[')
            temp = temp.apply(lambda x: x[0])
            temp = temp.str.split('(')
            temp = temp.apply(lambda x: x[0])
            temp = temp.str.strip(' )([].')
        additional_info[attribute_name] = temp

    if isolate_last_digits:
        temp_digits = additional_info[attribute_name].str.split(r'\D')
        temp_digits = temp_digits.apply(lambda x: x[-1])
        temp_digits = temp_digits[temp_digits.str.isnumeric()]
        additional_info[attribute_name] = temp_digits.astype(int)

    return movie_data.merge(additional_info, how='left', right_index=True, left_index=True)
