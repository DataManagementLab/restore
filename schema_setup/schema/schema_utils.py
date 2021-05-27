import hashlib

from schema_setup.incomplete_schema_setup.removal_method import RemovalMethod


def extend_by_rels(dataset, relationships, how='left'):
    ev_join_required = True
    while ev_join_required:
        ev_join_required = False
        for ev_r in relationships:
            if ev_r.incoming_table not in dataset.tables and ev_r.outgoing_table in dataset.tables:
                dataset = dataset.join(ev_r.incoming_table.incomplete_dataset, ev_r, how=how)

            if ev_r.outgoing_table not in dataset.tables and ev_r.incoming_table in dataset.tables:
                dataset = dataset.join(ev_r.outgoing_table.incomplete_dataset, ev_r, how=how)
                ev_join_required = True
    return dataset


def return_dataset(table, incomplete=True, keep_tfs=False):
    if incomplete:
        if keep_tfs:
            return table.incomplete_dataset_with_tfs
        else:
            return table.incomplete_dataset
    return table.load()


def join_tables(join_relationships, relationships_ordered=False, incomplete_join=False, how='inner', keep_tfs=False):
    if not relationships_ordered:
        join_relationships = relationship_order(join_relationships)

    joined_tables = set()
    dataset_join = None
    for next_r in join_relationships:

        if dataset_join is None:
            ds_incoming = return_dataset(next_r.incoming_table, incomplete=incomplete_join, keep_tfs=keep_tfs)
            ds_outgoing = return_dataset(next_r.outgoing_table, incomplete=incomplete_join, keep_tfs=keep_tfs)
            dataset_join = ds_incoming.join(ds_outgoing, next_r, how=how)

            joined_tables.add(next_r.incoming_table)
            joined_tables.add(next_r.outgoing_table)
        else:
            other_table = next_r.incoming_table
            if next_r.incoming_table in joined_tables:
                other_table = next_r.outgoing_table

            dataset_join = dataset_join.join(return_dataset(other_table, incomplete=incomplete_join, keep_tfs=keep_tfs),
                                             next_r, how=how)
            joined_tables.add(other_table)
    return dataset_join


def relationship_order(join_relationships, start_relationship=None, return_tables=False, first_table=None):
    finished_joins = set()
    joined_tables = set()
    ordered_join_relationships = []

    while len(finished_joins) < len(join_relationships):
        next_r = None

        if len(joined_tables) == 0:
            next_r = list(join_relationships)[0]
            if start_relationship is not None:
                next_r = start_relationship
            elif first_table is not None:
                next_r = [r for r in join_relationships if
                          r.incoming_table == first_table or r.outgoing_table == first_table][0]

        for r in join_relationships:
            if r not in finished_joins:
                if r.incoming_table in joined_tables or r.outgoing_table in joined_tables:
                    next_r = r

        # stopping criterion
        if next_r is None:
            assert len(finished_joins) == len(join_relationships), "Not all relationships joined"
            break

        ordered_join_relationships.append(next_r)
        finished_joins.add(next_r)
        joined_tables.add(next_r.incoming_table)
        joined_tables.add(next_r.outgoing_table)

    if return_tables:
        # first_table = list(ordered_join_relationships[0].tables.difference(ordered_join_relationships[1].tables))[0]
        joined_tables = [first_table]
        for r in ordered_join_relationships:
            other_table = r.incoming_table
            if r.incoming_table in joined_tables:
                other_table = r.outgoing_table
            joined_tables.append(other_table)

        return joined_tables, ordered_join_relationships

    return ordered_join_relationships


def custom_bfs(start_tables, exit_condition=None, process_step=None, **kwargs):
    """
    Performs a breadth-first search over the schema graph
    :param start_tables:
    :param exit_condition:
    :param process_step:
    :param kwargs:
    :return:
    """
    visited_tables = set()
    queue = [(t, []) for t in start_tables]

    while queue:
        table, r_seq = queue.pop(0)
        if table.name in visited_tables:
            continue
        visited_tables.add(table.name)

        stop_recursion = None
        if process_step is not None:
            stop_recursion = process_step(table=table, r_seq=r_seq, visited_tables=visited_tables, **kwargs)

        if exit_condition is not None and exit_condition(table):
            return table, r_seq

        if stop_recursion is None or not stop_recursion:
            for r in table.outgoing_relationships:
                queue.append((r.incoming_table, [r] + r_seq))
            for r in table.incoming_relationships:
                queue.append((r.outgoing_table, [r] + r_seq))

    if exit_condition is not None:
        raise ValueError("Finish condition was never true")


def relationship_path(start_table, end_table):
    """
    Find a path of relations connecting both tables using bfs
    :param schema:
    :param set_table:
    :param removal_table:
    :return:
    """
    _, rel_path = custom_bfs([start_table], exit_condition=lambda t: t == end_table)

    return rel_path


def stable_hash(string):
    return hashlib.sha224(string.encode('utf-8')).hexdigest()


def table_short_name(table_name):
    return ''.join([c[:2] for c in table_name.split('_')])


def generate_combined_scenario_name(dataset, projected_tables, tf_removals, tf_keep_rates,
                                    tuple_removal_tables, tuple_removal_keep_rate, removal_methods,
                                    removal_attrs, removal_attr_values, removal_attr_biases, seed, cascading_deletes):
    proj_tables = list(projected_tables)
    proj_tables.sort()
    proj_tables = '_'.join([table_short_name(t) for t in projected_tables])

    removal_desc = '_'.join([removal_name(removal_attr, removal_attr_bias, removal_attr_value, removal_method)
                             for removal_attr, removal_attr_bias, removal_attr_value, removal_method in
                             zip(removal_attrs, removal_attr_biases, removal_attr_values, removal_methods)])

    if not cascading_deletes:
        removal_desc += '_no_casc_del'

    tf_rem = '_'.join([f'{tf}_{keep_rate:.2f}' for tf, keep_rate in zip(tf_removals, tf_keep_rates)])

    scenario_name = f'{dataset}_{tf_rem}_{"_".join([table_short_name(t) for t in tuple_removal_tables])}_'
    scenario_name += "_".join([f'{r:.2f}' for r in tuple_removal_keep_rate])
    scenario_name += f'_{proj_tables}_{removal_desc}_s{seed}'
    return scenario_name


def removal_name(removal_attr, removal_attr_bias, removal_attr_value, removal_method):
    removal_desc = str(removal_method)
    if removal_method == RemovalMethod.BIAS or removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:
        removal_desc += f'{removal_attr}_{removal_attr_bias:.1f}'
        if removal_method == RemovalMethod.CATEGORICAL_PROB_BIAS:
            removal_desc += '_' + removal_attr_value
    return removal_desc


def validate_args(schema, args):
    assert all([schema.table_dict.get(t) is not None for t in args.tuple_removal_table])
    if args.completable_tables is not None:
        assert all([schema.table_dict.get(t) is not None for t in args.completable_tables])
    if args.fixed_completion_path is not None:
        assert all([schema.table_dict.get(t) is not None for t in args.fixed_completion_path])
    attribute_union = set()
    for t in schema.tables:
        attribute_union.update([a.full_name for a in t.attributes])
    assert all([ra in attribute_union or ra == 'none' for ra in args.removal_attr])
    assert all([tf in attribute_union or tf == 'none' for tf in args.tf_removals])

    assert args.tf_removals is None or len(args.tf_removals) == len(args.tf_keep_rates)
    assert len(args.tuple_removal_keep_rate) == len(args.tuple_removal_table) == len(args.removal_method) == \
           len(args.removal_attr) == len(args.removal_attr_bias)
    assert args.removal_attr_values is None or len(args.removal_attr_values) == len(args.removal_method)
