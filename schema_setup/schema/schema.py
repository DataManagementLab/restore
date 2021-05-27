class Schema:
    def __init__(self, default_separator=None):
        self.default_separator = default_separator
        self.tables = set()
        self.table_dict = dict()
        self.relationships = []

    def add_relationship(self, outgoing_table_name, outgoing_attribute_names, incoming_table_name,
                         incoming_attribute_names):
        incoming_attributes, incoming_table, outgoing_attributes, outgoing_table = self.find_relationship_objects(
            incoming_attribute_names, incoming_table_name, outgoing_attribute_names, outgoing_table_name)

        relationship = Relationship(self, outgoing_attributes, incoming_attributes)
        self.relationships.append(relationship)
        outgoing_table.outgoing_relationships.append(relationship)
        incoming_table.incoming_relationships.append(relationship)

    def find_relationship_objects(self, incoming_attribute_names, incoming_table_name, outgoing_attribute_names,
                                  outgoing_table_name):
        assert outgoing_table_name in self.table_dict.keys(), f"{outgoing_table_name} not defined as table"
        assert incoming_table_name in self.table_dict.keys(), f"{incoming_table_name} not defined as table"
        outgoing_table = self.table_dict[outgoing_table_name]
        incoming_table = self.table_dict[incoming_table_name]
        assert len(outgoing_attribute_names) == len(incoming_attribute_names), "Must be the same number of attributes"
        for outgoing_attribute in outgoing_attribute_names:
            assert outgoing_attribute in outgoing_table.attribute_dict.keys(), \
                f"{outgoing_attribute} not defined as attribute"
        for incoming_attribute in incoming_attribute_names:
            assert incoming_attribute in incoming_table.attribute_dict.keys(), \
                f"{incoming_attribute} not defined as attribute"
        outgoing_attributes = [outgoing_table.attribute_dict[outgoing_attribute] for outgoing_attribute in
                               outgoing_attribute_names]
        incoming_attributes = [incoming_table.attribute_dict[incoming_attribute] for incoming_attribute in
                               incoming_attribute_names]
        return incoming_attributes, incoming_table, outgoing_attributes, outgoing_table


class Table:
    def __init__(self, schema, name, attribute_names, primary_key=None, irrelevant_attribute_names=None, filename=None,
                 separator=None, sample_rate=1.0, table_size=0.0):
        self.name = name
        self.schema = schema
        self.sample_rate = sample_rate
        self.table_size = table_size

        # for csv files
        self.filename = filename
        self.separator = separator
        if self.separator is None and schema.default_separator is not None:
            self.separator = schema.default_separator

        # store attributes as list of strings
        self._csv_columns = attribute_names
        if irrelevant_attribute_names is None:
            irrelevant_attribute_names = []
        self._irrelevant_csv_columns = irrelevant_attribute_names

        # store attributes as objects
        self.attributes = []
        self.primary_key = []
        self.attribute_dict = {}
        for attribute_name in attribute_names:
            if attribute_name not in irrelevant_attribute_names:
                attribute = Attribute(self, attribute_name)
                self.attribute_dict[attribute_name] = attribute
                self.attributes.append(attribute)
                if primary_key is not None and attribute_name in primary_key:
                    self.primary_key.append(attribute)
                    attribute.is_pk = True

        self.incoming_relationships = []
        self.outgoing_relationships = []

        schema.table_dict[name] = self

    @property
    def full_csv_columns(self):
        return [self.name + '.' + attribute_name for attribute_name in self._csv_columns]

    @property
    def full_irrelevant_csv_columns(self):
        return [self.name + '.' + attribute_name for attribute_name in self._irrelevant_csv_columns]

    def __str__(self):
        return self.name


class Attribute:
    def __init__(self, table, name, is_pk=False, is_tf=False, is_fk=False, is_current_tf=False):
        self.table = table
        self.name = name

        self.categorical_columns_dict = None
        self.meta_type = None
        self.null_value = None

        self.is_pk = is_pk
        self.is_tf = is_tf
        self.is_fk = is_fk
        self.is_current_tf = is_current_tf

    @property
    def full_name(self):
        if self.table is not None:
            return self.table.name + '.' + self.name
        return self.name

    def __str__(self):
        return self.full_name


class Relationship:
    def __init__(self, schema, outgoing_attributes, incoming_attributes):
        self.schema = schema
        self.outgoing_table = outgoing_attributes[0].table
        self.outgoing_attributes = outgoing_attributes
        for outgoing_attribute in outgoing_attributes:
            outgoing_attribute.is_fk = True

        self.incoming_table = incoming_attributes[0].table
        self.incoming_attributes = incoming_attributes

        self.tf_attribute = Attribute(self.incoming_table, 'tf_' + '_'.join(
            [outgoing_attribute.full_name for outgoing_attribute in outgoing_attributes]), is_tf=True)
        self.incoming_table.attribute_dict[self.tf_attribute.full_name] = self.tf_attribute

        self.incoming_table.attributes.append(self.tf_attribute)
        self.tuple_factor_name = self.tf_attribute.full_name

    def __str__(self):
        return self.incoming_table.name + '-' + self.outgoing_table.name

    @property
    def identifier(self):
        return self.tuple_factor_name

    @property
    def tables(self):
        return {self.incoming_table, self.outgoing_table}
