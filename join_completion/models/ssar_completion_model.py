import logging
from copy import copy

import numpy as np

from join_completion.models.flat_ar_completion_model import FlatARCompletionModel
from schema_setup.schema.schema_utils import join_tables, extend_by_rels
from ssar.common import CsvTable, Discretize
from ssar.hierarchy_dataloader import HierarchyNode
from ssar.train_model import train_autoregressive

logger = logging.getLogger(__name__)


class SSARHierarchyNode(HierarchyNode):
    def __init__(self, id_scope, scopes, max_hierarchy_tuples):
        HierarchyNode.__init__(self, id_scope, scopes, max_hierarchy_tuples=max_hierarchy_tuples)
        self.hierarchy_attributes = None
        self.hierarchy_rs = None
        self.table = None
        self.column_id_name = None

    def map_to_validation_schema(self, r_mapping, a_mapping):
        n = copy(self)
        if self.hierarchy_attributes is not None:
            n.hierarchy_attributes = [a_mapping[a] for a in self.hierarchy_attributes]
        if self.hierarchy_rs is not None:
            n.hierarchy_rs = [r_mapping[r] for r in self.hierarchy_rs]
        n.children = [c.map_to_validation_schema(r_mapping, a_mapping) for c in self.children]
        for c in n.children:
            c.parent = n
        return n

    def __str__(self):
        return f'Hierarchy({", ".join([str(a) for a in self.hierarchy_attributes])})'


class SSARCompletionModel(FlatARCompletionModel):
    def __init__(self, model_directory, r=None, inverse=False, params=None):
        FlatARCompletionModel.__init__(self, model_directory, r=r, inverse=inverse, params=params)
        self.hierarchies = []
        self.set_r = r

    def train(self):
        full_join_dataset, table = self._training_data()

        for h in self.hierarchies:
            h_ev = join_tables(h.hierarchy_rs, relationships_ordered=True, incomplete_join=True, how='left')
            h_ev = h_ev.project(h.hierarchy_attributes)

            h_table = CsvTable(None, h_ev.df_rows, h_ev.df_rows.columns, {})
            for c in h_table.columns:
                # check if collision with table, if so remove
                for coll_c in table.columns:
                    if coll_c.name == c.name:
                        # overwrite the distinct values s.t. we really reuse the embeddings later on
                        c.all_distinct_values = coll_c.all_distinct_values
            h.table = h_table

        hierarchy_top_ids = []
        for h in self.hierarchies:
            ids = full_join_dataset.df_rows[h.hierarchy_attributes[0].full_name].values
            hierarchy_top_ids.append(Discretize(h.table.columns[0], data=ids))

        hierarchy_excluded_ids_per_node = []
        for h in self.hierarchies:
            excluded_ids_per_node = dict()
            for node_id, n in h.node_dict.items():
                id_col_name = h.table.columns[n.id_scope].name
                if id_col_name not in full_join_dataset.df_rows.columns:
                    excluded_ids_per_node = None
                    break

                ids = full_join_dataset.df_rows[id_col_name].values
                excluded_ids_per_node[node_id] = Discretize(h.table.columns[n.id_scope], data=ids)
            hierarchy_excluded_ids_per_node.append(excluded_ids_per_node)

        acc, training_time, model = train_autoregressive(table, self.model_directory, self.model_name,
                                                         hierarchies=self.hierarchies,
                                                         hierarchy_top_ids=hierarchy_top_ids,
                                                         hierarchy_tables=[h.table for h in self.hierarchies],
                                                         hierarchy_excluded_ids_per_node=hierarchy_excluded_ids_per_node,
                                                         **self.params)
        self.model = model
        return training_time, acc

    def map_to_validation_schema(self, t_mapping, r_mapping, a_mapping):
        mapped_model = super().map_to_validation_schema(t_mapping, r_mapping, a_mapping)

        mapped_model.hierarchies = [h.map_to_validation_schema(r_mapping, a_mapping) for h in self.hierarchies]
        mapped_model.set_r = r_mapping[self.set_r]

        return mapped_model

    def expand_hierarchies(self, max_hierarchy_depth, self_evidence_only):
        evidence_tables = self.evidence_tables
        for ev_t in evidence_tables:
            for ev_r in ev_t.incoming_relationships:
                if self.set_r == ev_r or ev_r.outgoing_table not in evidence_tables:
                    if self_evidence_only and not self.set_r == ev_r:
                        continue

                    def expand_hierarchy(t, parent_node, parent_r, hierarchy_attributes, hierarchy_rs, expand_set_r,
                                         ar_tables, visited_tables, depth, max_hierarchy_depth):
                        if depth > max_hierarchy_depth:
                            return
                        if t in visited_tables:
                            return
                        if expand_set_r and t not in ar_tables:
                            return

                        hierarchy_rs.append(parent_r)

                        # first search all neighbors without fanout
                        stage_tables = {t}
                        queue = [t]
                        while queue:
                            t = queue.pop()
                            for r in t.outgoing_relationships:
                                if r.incoming_table not in visited_tables:
                                    if expand_set_r and r.incoming_table not in ar_tables:
                                        continue
                                    stage_tables.add(r.incoming_table)
                                    queue.append(r.incoming_table)
                                    hierarchy_rs.append(r)
                        visited_tables.update(stage_tables)

                        scopes = []
                        id_scope = -1
                        for st in stage_tables:
                            for a in st.incomplete_dataset.attributes:
                                if a.is_fk:
                                    continue
                                if a.is_pk:
                                    if a != t.primary_key[0]:
                                        continue
                                    hierarchy_attributes.append(a)
                                    id_scope = len(hierarchy_attributes) - 1
                                    continue

                                hierarchy_attributes.append(a)
                                scopes.append(len(hierarchy_attributes) - 1)

                        ssar_node = SSARHierarchyNode(id_scope=id_scope, scopes=scopes,
                                                      max_hierarchy_tuples=self.params['max_hierarchy_tuples'])
                        ssar_node.parent = parent_node
                        parent_node.children.append(ssar_node)

                        for st in stage_tables:
                            for r in st.incoming_relationships:
                                expand_hierarchy(r.outgoing_table, ssar_node, r, hierarchy_attributes, hierarchy_rs,
                                                 expand_set_r, ar_tables, visited_tables, depth + 1,
                                                 max_hierarchy_depth)

                        return ssar_node

                    hierarchy_attributes = [ev_t.primary_key[0]]
                    hierarchy_rs = []
                    root_node = SSARHierarchyNode(id_scope=0, scopes=[],
                                                  max_hierarchy_tuples=self.params['max_hierarchy_tuples'])
                    expand_hierarchy(ev_r.outgoing_table, root_node, ev_r, hierarchy_attributes, hierarchy_rs,
                                     expand_set_r=self.set_r == ev_r, ar_tables=self.tables, visited_tables={ev_t},
                                     depth=0, max_hierarchy_depth=max_hierarchy_depth)
                    root_node.hierarchy_attributes = hierarchy_attributes
                    root_node.hierarchy_rs = hierarchy_rs
                    root_node.assign_depths()
                    root_node.assign_node_dict()
                    self.hierarchies.append(root_node)

    def expand_set_evidence(self, table, completion_r):
        for r in table.outgoing_relationships:
            if r in completion_r:
                continue
            completion_r.append(r)
            self.expand_set_evidence(r.outgoing_table, completion_r)
        return completion_r

    def transform_hierarchy_evidence(self, current_join):
        hierarchy_batches = []
        for h in self.hierarchies:
            ids = current_join.df_rows[h.hierarchy_attributes[0].full_name].values
            assigned_parent_ids = Discretize(h.table.columns[0], data=ids)

            hierarchy_evidence = extend_by_rels(current_join, h.hierarchy_rs, how='inner')
            hierarchy_evidence = hierarchy_evidence.project(h.hierarchy_attributes)
            hierarchy_evidence = self.transform_to_evidence(hierarchy_evidence.df_rows.values, columns=h.table.columns)
            hierarchy_batches.append(h.transform_tensors(hierarchy_evidence, assigned_parent_ids=assigned_parent_ids,
                                                         shuffle_set_members=False))

        return hierarchy_batches

    def reshape_hierarchy_evidence(self, hierarchy_batches, additional_tuples=None, sample_idx=None):
        assert additional_tuples is not None or sample_idx is not None
        for hbs, h in zip(hierarchy_batches, self.hierarchies):
            for node_id, h_evidence in hbs.items():
                len_cols = h_evidence.shape[1]
                h_evidence = h_evidence.reshape(-1, h.max_hierarchy_tuples ** h.node_dict[node_id].depth, len_cols)
                if additional_tuples is not None:
                    h_evidence = np.repeat(h_evidence, additional_tuples.astype(int), axis=0)
                else:
                    h_evidence = h_evidence[sample_idx]
                h_evidence = h_evidence.reshape(-1, len_cols)

                hbs[node_id] = h_evidence

        return hierarchy_batches

    def __str__(self):
        str_rep = 'SSARModel(relations=' + ','.join([str(t) for t in self.table_ordering()]) + ')'
        for cr in self.completion_relationships:
            str_rep += ('\n\t' + str(cr))
        for hierarchy in self.hierarchies:
            str_rep += ('\n\t' + str(hierarchy))
        return str_rep
