import logging

import networkx as nx

from join_completion.models.completion_model import CompletionSetup
from join_completion.models.flat_ar_completion_model import FlatARCompletionModel
from join_completion.models.ssar_completion_model import SSARCompletionModel
from schema_setup.schema.schema_utils import custom_bfs

logger = logging.getLogger(__name__)


def merge_flat_ar(model, other_model):
    if model == other_model:
        return None, False

    if not type(model) == FlatARCompletionModel or not type(other_model) == FlatARCompletionModel:
        return None, False

    # hierarchy of evidence tables
    for cs in model.completion_relationships:
        for other_cs in other_model.completion_relationships:
            if not ((cs.evidence_tables.issubset(other_cs.evidence_tables) and cs.tables.issubset(other_cs.tables)) or
                    (other_cs.evidence_tables.issubset(cs.evidence_tables) and other_cs.tables.issubset(cs.tables))):
                return None, False

    potential_merge_graph = nx.compose(model.r_graph, other_model.r_graph)
    assert nx.is_directed_acyclic_graph(potential_merge_graph)

    assert model.model_directory == other_model.model_directory
    merged = FlatARCompletionModel(model.model_directory)
    merged.r_graph = potential_merge_graph
    merged.completion_relationships = model.completion_relationships.union(other_model.completion_relationships)
    assert len(merged.completion_relationships) >= 2
    return merged, True


def find_flat_ar_models(schema, model_directory=None, params=None, completable_tables=None,
                        fixed_completion_path=None):
    """
    Suggests flat AR models such that any join/single table can be completed
        (1) Find all relationships where a completion might be necessary
            - BFS on graph starting from complete table
            - If we encounter a relationship that has to be completed, we greedily expand the evidence tables
                (e.g., for orders->orderline, they would also require the customer table as evidence)
        (2) Merge Flat AR models
            - Condition for merging: for each completion setup of a model, the evidence tables are a subset, just like
                all tables (e.g., customer->order could be merged into customer->order->orderline)
    :param completable_tables: Single tables for which we want a completion. If None, all queries are supported.
        This requires additional models to be learned. For instance, for
            comp[c]->movie_comp[ic]<-movie[ic]->movie_dir[ic]<-director[c]
        we would need only two models instead of four (the difference is: having joined comp, movie_comp, movie, we do
        not need a model for the direction movie->movie_dir<-director if we do not have to support joins).
    :param schema:
    :param model_directory:
    :param params:
    :return:
    """
    ar_models = []

    # find all completion setups we need
    if completable_tables is None:
        required_completion_setups = find_required_completion_setups(schema, fixed_completion_path)
    else:
        required_completion_setups = find_required_completion_setups_single_table(schema, completable_tables,
                                                                                  fixed_completion_path)

    for r, inv in required_completion_setups:
        ar_models.append(FlatARCompletionModel(model_directory, r=r, inverse=inv))

    # merge as many models as possible
    merged_models = []
    merge_possible = True
    while merge_possible:
        merge_possible = False
        for i, m1 in enumerate(ar_models):
            if m1.merged:
                continue
            for m2 in ar_models:
                if m2.merged or m1 == m2:
                    continue
                model, mergeable = merge_flat_ar(m1, m2)
                if mergeable:
                    m1.merged = True
                    m2.merged = True
                    merge_possible = True
                    merged_models.append(model)
        ar_models = merged_models + [m for m in ar_models if not m.merged]
        merged_models = []

    for ar_model in ar_models:
        ar_model.params = params
        logger.info(str(ar_model))

    return ar_models


def find_required_completion_setups(schema, fixed_completion_path):
    required_completion_setups = set()
    for complete_table in schema.tables:
        # bfs in schema using this table
        if complete_table.complete:
            visited_tables = set()
            queue = [complete_table]

            while queue:
                table = queue.pop(0)
                if table.name in visited_tables:
                    continue
                visited_tables.add(table)

                # table orderline, outgoing_relationships=[orderline->order]
                for r in table.outgoing_relationships:
                    assert r.incoming_table != table
                    if r.incoming_table in visited_tables:
                        continue
                    # orders incomplete or additionally generated orderlines
                    if not (r.incoming_table.complete and table.complete and len(r.pks_without_fk) == 0):
                        required_completion_setups.add((r, True))
                        assert CompletionSetup(set(), r, set(), True).evidence_table == table
                    queue.append(r.incoming_table)

                # table order, incoming=[orderline->order]
                for r in table.incoming_relationships:
                    assert r.outgoing_table != table
                    if r.outgoing_table in visited_tables:
                        continue
                    # orders incomplete or additionally generated orderlines
                    if not (r.outgoing_table.complete and table.complete and len(r.pks_without_fk) == 0):
                        required_completion_setups.add((r, False))
                        assert CompletionSetup(set(), r, set(), False).evidence_table == table

                    # matching required, we do this using LSH, so this is no longer required
                    # if len(r.pks_without_fk) > 0:
                    #     required_completion_setups.add((r, True))
                    queue.append(r.outgoing_table)

    required_completion_setups = reduce_to_completion_path(fixed_completion_path, required_completion_setups)

    return required_completion_setups


def reduce_to_completion_path(fixed_completion_path, required_completion_setups):
    if fixed_completion_path is not None:
        required_completion_setups = [(r, inv) for r, inv in required_completion_setups if
                                      r.incoming_table.name in fixed_completion_path and
                                      r.outgoing_table.name in fixed_completion_path]

    else:
        required_completion_setups = list(required_completion_setups)
        
    return required_completion_setups


def find_required_completion_setups_single_table(schema, completable_tables, fixed_completion_path):
    required_completion_setups = set()

    # find all possible completion paths for all incomplete tables
    for t in schema.tables:
        # not necessary, just check if among completable tables
        # if t.complete:
        #     continue

        if t.name not in completable_tables:
            continue

        def find_completion_paths(table=None, r_seq=None, required_completion_setups=None, **kwargs):
            if table.complete:
                path_tables = {table}
                for r in r_seq:
                    if r.incoming_table in path_tables:
                        inv = False
                    elif r.outgoing_table in path_tables:
                        inv = True
                    else:
                        raise NotImplementedError

                    required_completion_setups.add((r, inv))
                    assert CompletionSetup(set(), r, set(), inv).evidence_table in path_tables
                    path_tables.update(r.tables)

                return True
            return False

        custom_bfs({t}, process_step=find_completion_paths, required_completion_setups=required_completion_setups)

    required_completion_setups = reduce_to_completion_path(fixed_completion_path, required_completion_setups)

    # add anything in the completable tables
    for r in schema.relationships:
        if r.incoming_table.name in completable_tables and r.outgoing_table.name in completable_tables:
            if (r, True) not in required_completion_setups:
                required_completion_setups.append((r, True))
            if (r, False) not in required_completion_setups:
                required_completion_setups.append((r, False))

    return required_completion_setups


def find_ssar_models(schema, model_directory=None, params=None, completable_tables=None, fixed_completion_path=None):
    """
    In an ensemble of flat AR models find which set AR models can be learned in addition
    :param schema:
    :param model_directory:
    :param params:
    :return:
    """
    flat_ar_models = find_flat_ar_models(schema, model_directory=model_directory, params=params,
                                         completable_tables=completable_tables,
                                         fixed_completion_path=fixed_completion_path)
    required_ar_models = []
    for m in flat_ar_models:

        # for one flat model we need a set model for every non-inverse relationship (e.g., customer->order)
        # if there are inverse relationships after this set, we can handle these as well (we cannot handle them if they
        #   are before because in this case they should not see the set evidence and we have no masking for this case)
        inverse_completion_relationships = [cs_r for cs_r in m.completion_relationships if cs_r.inverse]
        non_inverse_completion_relationships = [cs_r for cs_r in m.completion_relationships if not cs_r.inverse]

        # for every non_inverse_completion_relationships we can potentially learn a set model
        handled_inverse_completion_relationships = set()
        if len(non_inverse_completion_relationships) > 0:
            for set_cs_r in non_inverse_completion_relationships:
                # later on the AR model has to be able to perform all the completions. This restricts the variable
                # ordering for the AR model (evidence relationships before r before completion_relationships). We
                # encode this a directed graph.
                set_model = configure_ssar_model(handled_inverse_completion_relationships,
                                                 inverse_completion_relationships,
                                                 m, model_directory, params, set_cs_r.r)
                required_ar_models.append(set_model)

        # if there are no non_inverse_completion_relationships we might be able to learn a set model if there is a
        # non-inverse relationship in the evidence
        else:
            for cs_r in inverse_completion_relationships:
                if cs_r in handled_inverse_completion_relationships:
                    continue

                # this can be our set relationship
                if len(cs_r.evidence_relationships) > 0:
                    # preferred set completion relationship is the one next to evidence
                    preferred_set_rels = [r for r in cs_r.evidence_relationships if
                                          r.outgoing_table == cs_r.evidence_table]
                    assert len(preferred_set_rels) == 1
                    set_r = preferred_set_rels[0]

                    set_model = configure_ssar_model(handled_inverse_completion_relationships,
                                                     inverse_completion_relationships,
                                                     m, model_directory, params, set_r)
                    required_ar_models.append(set_model)

        if not len(handled_inverse_completion_relationships) == len(inverse_completion_relationships):
            # add hierarchy method as below!
            ssar_model = SSARCompletionModel(model_directory, params=params)
            ssar_model.completion_relationships = m.completion_relationships
            ssar_model.r_graph = m.r_graph
            ssar_model.expand_hierarchies(params['max_hierarchy_depth'], params['self_evidence_only'])
            required_ar_models.append(ssar_model)

    logger.info("Required set/flat AR models")
    for ar_model in required_ar_models:
        ar_model.params = params
        logger.info(str(ar_model))

    return required_ar_models


def configure_ssar_model(handled_inverse_completion_relationships, inverse_completion_relationships, m, model_directory,
                         params, set_r):
    ssar_model = SSARCompletionModel(model_directory, r=set_r, inverse=False, params=params)

    potential_merge_graph = nx.compose(m.r_graph, ssar_model.r_graph)
    assert nx.is_directed_acyclic_graph(potential_merge_graph)

    set_cs_r = list(ssar_model.completion_relationships)[0]

    handled_inverse_rels = [cs_r for cs_r in inverse_completion_relationships
                            if cs_r.evidence_tables.issuperset(set_cs_r.evidence_tables)]
    handled_inverse_completion_relationships.update(handled_inverse_rels)

    ssar_model.completion_relationships = handled_inverse_rels + [set_cs_r]
    ssar_model.r_graph = potential_merge_graph

    ssar_model.expand_hierarchies(params['max_hierarchy_depth'], params['self_evidence_only'])

    return ssar_model
