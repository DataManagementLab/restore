class PlottingQuery:
    def __init__(self, id, attribute, aggregation, aqp_query, general_value_dict):
        self.id = id
        self.attribute = attribute
        self.aggregation = aggregation
        self.aqp_query = aqp_query
        self.sql_string = aqp_query.sql_string(aggregation, cat_value_dict=general_value_dict)
        self.completion_tables = set(aqp_query.completion_tables)
        self.groupings = set(aqp_query.grouping_attributes)

    def rename_hosts(self, sql_string):
        # airbnb vs. landlord dataset
        sql_string = sql_string.replace('listings', 'apartment')
        sql_string = sql_string.replace('hosts', 'landlord')
        sql_string = sql_string.replace('host', 'landlord')
        return sql_string

    def paper_formatted_sql_string(self):
        sql_string = self.sql_string
        # replace codes by actual categorical value
        sql_string = self.rename_hosts(sql_string)
        renamed_completion_tables = [self.rename_hosts(t) for t in self.completion_tables]

        joins = ' NATURAL JOIN '.join(renamed_completion_tables)
        sql_string = sql_string.replace('WHERE', f'FROM {joins} WHERE')

        for t in renamed_completion_tables:
            sql_string = sql_string.replace(f'{t}.', '')

        sql_string = sql_string.replace('>=', '$\ge$')
        sql_string = sql_string.replace('<=', '$\le$')
        sql_string = sql_string.replace('_', '\_')

        return sql_string