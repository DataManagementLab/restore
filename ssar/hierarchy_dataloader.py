import logging
import math
from random import randint
from time import perf_counter

import numpy as np
import torch
from numba import njit, types, prange
from numba.typed import Dict

from ssar.utils import in_set

logger = logging.getLogger(__name__)


@njit()
def assign_child_tuples(max_hierarchy_tuples, assigned_set_ids, set_ids, shuffle_set_members=True, excluded_ids=None,
                        projected_ids=None, data=None):
    set_id_bins = Dict.empty(
        key_type=types.int64,
        value_type=types.int64,
    )
    # initialize s.t. numba understands the types
    set_id_mapping_list = [[np.int(x) for x in range(0)] for i in range(0)]

    for i in range(len(set_ids)):
        set_id = set_ids[i]
        # means set id is nan
        if set_id == 0:
            continue

        if set_id_bins.get(set_id) is None:
            set_id_bins[set_id] = len(set_id_bins)
            set_id_mapping_list.append([np.int(x) for x in range(0)])
        set_id_mapping_list[set_id_bins[set_id]].append(i)

    tensor = np.zeros((len(assigned_set_ids) * max_hierarchy_tuples, data.shape[1] + 1), dtype=np.float32)
    ids = np.zeros(len(assigned_set_ids) * max_hierarchy_tuples)

    for i in prange(len(assigned_set_ids)):
        set_id = assigned_set_ids[i]

        # can happen if there is no set evidence at all for this tuple or set_id is Nan
        if set_id_bins.get(set_id) is None:
            continue

        other_set_idx = set_id_mapping_list[set_id_bins[set_id]]

        if shuffle_set_members:
            num_other_set_tuples = randint(0, min(len(other_set_idx), max_hierarchy_tuples))
            # shuffled_other_set_idx = list(np.random.choice(np.asarray(other_set_idx), size=num_other_set_tuples))
            shuffled_other_set_idx = [other_set_idx[randint(0, len(other_set_idx) - 1)] for _ in
                                      range(num_other_set_tuples)]

        else:
            shuffled_other_set_idx = other_set_idx[:max_hierarchy_tuples]

        if excluded_ids is not None:
            shuffled_other_set_idx = [x for x in shuffled_other_set_idx if projected_ids[x] != excluded_ids[i]]

        shuffled_other_set_idx = np.asarray(shuffled_other_set_idx)

        set_start_idx = i * max_hierarchy_tuples
        tensor[set_start_idx:set_start_idx + len(shuffled_other_set_idx), 1:] = data[shuffled_other_set_idx]
        tensor[set_start_idx:set_start_idx + len(shuffled_other_set_idx), 0] = 1
        ids[set_start_idx:set_start_idx + len(shuffled_other_set_idx)] = projected_ids[shuffled_other_set_idx]

    return tensor, ids


class HierarchyNode:
    NO_NODES = 0

    def __init__(self, id_scope, scopes, children=None, node_id=None, max_hierarchy_tuples=10, depth=0):
        if node_id is not None:
            self.node_id = node_id
        else:
            # assign unique id to node in the hierarchy
            self.node_id = HierarchyNode.NO_NODES
            HierarchyNode.NO_NODES += 1

        self.id_scope = id_scope
        self.scopes = scopes
        self.parent = None
        self.max_hierarchy_tuples = max_hierarchy_tuples
        self.children = children
        self.depth = depth
        if self.children is None:
            self.children = []
        self.node_dict = dict()

        for c in self.children:
            c.parent = self

    def assign_node_dict(self, nd=None):
        if nd is None:
            nd = self.node_dict
        nd[self.node_id] = self
        for c in self.children:
            c.assign_node_dict(nd=nd)

    def assign_depths(self, depth=0):
        self.depth = depth
        for c in self.children:
            c.assign_depths(depth=depth + 1)

    def root_tensor(self, data):
        ids = data[:, self.id_scope]
        # if blowup:
        #     tensor = data[self.scopes].reshape(-1, len(self.scopes))
        _, idx = np.unique(ids, return_index=True)
        # make sure to preserve the order
        idx = np.sort(idx)
        ids = ids[idx]
        tensor = data[idx, self.scopes].reshape(len(idx), len(self.scopes))

        return tensor, ids

    def inner_tensor(self, assigned_parent_ids, data, shuffle_set_members, excluded_ids_per_node):
        # this does not work since tuples in a set might become incomplete
        # _, idx = np.unique(data[:, self.id_scope], return_index=True)
        # idx = idx[~np.isnan(data[idx, self.id_scope])]
        projected_ids = data[:, self.id_scope].astype(int)
        projected_parent_ids = data[:, self.parent.id_scope].astype(int)
        data = data[:, self.scopes].reshape(data.shape[0], len(self.scopes))

        excluded_ids = None
        if excluded_ids_per_node is not None:
            excluded_ids = excluded_ids_per_node[self.node_id].astype(int)
            assert len(excluded_ids) == len(assigned_parent_ids)

        assert not np.any(np.isnan(assigned_parent_ids))
        assert not np.any(np.isnan(projected_parent_ids))

        start_t = perf_counter()
        tensor, ids = assign_child_tuples(self.max_hierarchy_tuples, assigned_parent_ids.astype(int),
                                          projected_parent_ids, shuffle_set_members=shuffle_set_members,
                                          excluded_ids=excluded_ids, projected_ids=projected_ids, data=data)
        logger.info(f"Assigning inner tensor took {perf_counter() - start_t:.2f}s")
        return tensor, ids

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def transform_tensors(self, data, shuffle_set_members=True, tensor_dict=None,
                          assigned_parent_ids=None, excluded_ids_per_node=None):

        # root node of hierarchy
        if self.is_root:
            tensor_dict = dict()
            if assigned_parent_ids is None:
                _, ids = self.root_tensor(data)
            else:
                ids = assigned_parent_ids

        # some inner node or child
        else:
            tensor, ids = self.inner_tensor(assigned_parent_ids, data, shuffle_set_members, excluded_ids_per_node)
            tensor_dict[self.node_id] = tensor

        for c in self.children:
            c.transform_tensors(data, shuffle_set_members=shuffle_set_members, tensor_dict=tensor_dict,
                                assigned_parent_ids=ids, excluded_ids_per_node=excluded_ids_per_node)

        return tensor_dict

    def sample_batches(self, data, root_ids, batch_size=10, excluded_ids_per_node=None):
        # only consider relevant data where the root_ids are present
        start_t = perf_counter()
        in_sample = in_set(data[:, self.id_scope], root_ids)
        data = data[in_sample]

        tensor_dict = self.transform_tensors(data, shuffle_set_members=True, assigned_parent_ids=root_ids,
                                             excluded_ids_per_node=excluded_ids_per_node)

        tensor_dict = self.split_batches(batch_size, math.ceil(len(root_ids) / batch_size), tensor_dict)
        # transform to torch tensors
        tensor_dict = {node_id: [torch.from_numpy(b) for b in batches] for node_id, batches in tensor_dict.items()}
        logger.info(f"Total batch sampling for hierarchy took {perf_counter() - start_t:.2f}s")
        return tensor_dict

    def split_batches(self, batch_size, no_batches, tensor_dict):
        batches = {node_id: np.split(evidence,
                                     (np.arange(1, no_batches) * batch_size) * (self.max_hierarchy_tuples
                                                                                ** self.node_dict[node_id].depth))
                   for node_id, evidence in tensor_dict.items()}
        return batches
