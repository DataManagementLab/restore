from schema_setup.schema.schema import Schema, Table


def gen_synthetic_schema(no_tuples=1000000, tf_constant=5):
    schema = Schema(default_separator=';')
    schema.tables.add(Table(schema, 'complete',
                            ['id', 'attribute_a'], filename='complete.csv',
                            primary_key=['id'], table_size=no_tuples))
    schema.tables.add(Table(schema, 'incomplete',
                            ['id', 'attribute_b', 'complete_id'], filename='incomplete.csv',
                            primary_key=['id'], table_size=no_tuples * tf_constant))
    schema.add_relationship('incomplete', ['complete_id'], 'complete', ['id'])
    return schema
