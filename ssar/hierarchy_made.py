"""SetResMADE."""

import torch
import torch.nn as nn

from ssar.hierarchy_dataloader import HierarchyNode
from ssar.made import MaskedLinear, MADE, MaskedResidualBlock
from ssar.masked_blocks import PartialMaskedResidualBlock, PartialMaskedLinear


class DNNHierarchyNode(HierarchyNode):
    def __init__(self, id_scope, scopes, embedding_idxs, children, node_id, max_hierarchy_tuples, depth):
        HierarchyNode.__init__(self, id_scope, scopes, children=children, node_id=node_id,
                               max_hierarchy_tuples=max_hierarchy_tuples, depth=depth)
        # map to correct index of embedding of AR model
        self.embedding_idxs = embedding_idxs
        assert len(self.embedding_idxs) == len(self.scopes)

        self.encoder_network = []

    def initialize_dnn(self, input_encoded_dist_sizes, hierarchy_hidden_sizes, activation, encoder_networks):
        # create mapping node_id -> node for root node
        if self.is_root:
            assert len(self.children) <= 1, "Root node of a hierarchy must not have several children. Simply create " \
                                            "several hierarchies to express this"

            self.assign_node_dict()

        # root node does not need FF neural network since it is the top hierarchy (e.g., cust - order join we do not
        #   set pooling for customers, only for orders)
        else:

            # how many inputs to encode this step of the hierarchy
            nin = sum([input_encoded_dist_sizes[e_idx] for e_idx in self.embedding_idxs])
            # plus children encodings
            nin += len(self.children) * hierarchy_hidden_sizes[-1]

            hs = [nin] + hierarchy_hidden_sizes
            for h0, h1 in zip(hs, hs[1:]):
                self.encoder_network.extend([
                    nn.Linear(h0, h1),
                    activation(inplace=True),
                ])

            self.encoder_network = nn.Sequential(*self.encoder_network)
            encoder_networks.append(self.encoder_network)

        for c in self.children:
            c.initialize_dnn(input_encoded_dist_sizes, hierarchy_hidden_sizes, activation, encoder_networks)

        return encoder_networks

    def forward(self, enc_h_inputs, average_embd):
        # recurse over children
        child_enc = []
        for c in self.children:
            child_enc.append(c.forward(enc_h_inputs, average_embd))

        if self.is_root:
            return child_enc[0]

        # embed my sets
        sets_present, node_evidence = enc_h_inputs[self.node_id]
        child_enc.append(node_evidence)

        set_evidence = torch.cat(child_enc, 1)

        enc_set = self.encoder_network(set_evidence)
        enc_set = enc_set * sets_present[:, None]
        # "group by" sets, sum
        enc_set = enc_set.view((-1, self.max_hierarchy_tuples, enc_set.shape[-1]))
        enc_set = torch.sum(enc_set, axis=1)

        if average_embd:
            sets_present = sets_present.view(-1, self.max_hierarchy_tuples)
            sets_present = torch.sum(sets_present, axis=1)
            sets_present = 1 / torch.clamp(sets_present, min=1.0)
            enc_set = enc_set * sets_present[:, None]

        return enc_set


class HierarchyMADE(MADE):

    def __init__(
            self,
            nin,
            hidden_sizes,
            hierarchy_hidden_sizes,
            hierarchy_embedding_layer_idxs,
            nout,
            hierarchies,
            num_masks=1,
            natural_ordering=True,
            input_bins=None,
            additional_input_bins=None,
            activation=nn.ReLU,
            do_direct_io_connections=False,
            input_encoding=None,
            output_encoding='one_hot',
            embed_size=32,
            residual_connections=False,
            column_masking=False,
            seed=11123,
            fixed_ordering=None,
            average_embd=False,
            max_embedding_size=16384,
            priors=None
    ):
        super().__init__(nin, hidden_sizes, nout, num_masks=num_masks, natural_ordering=natural_ordering,
                         input_bins=input_bins, additional_input_bins=additional_input_bins, activation=activation,
                         do_direct_io_connections=do_direct_io_connections, input_encoding=input_encoding,
                         output_encoding=output_encoding, embed_size=embed_size,
                         residual_connections=residual_connections, column_masking=column_masking, seed=seed,
                         fixed_ordering=fixed_ordering, skip_net=True, max_embedding_size=max_embedding_size,
                         priors=priors)

        print('fixed_ordering', fixed_ordering, 'seed', seed,
              'natural_ordering', natural_ordering)
        self.hierarchy_hidden_sizes = hierarchy_hidden_sizes
        self.hierarchy_embedding_layer_idxs = hierarchy_embedding_layer_idxs
        self.hierarchies = hierarchies

        # hierarchy encoder networks
        hierarchy_encoding_size = len(self.hierarchies) * hierarchy_hidden_sizes[-1]
        self.encoder_networks = []
        for h in self.hierarchies:
            h.initialize_dnn(self.input_encoded_dist_sizes, hierarchy_hidden_sizes, activation, self.encoder_networks)
        self.encoder_networks = nn.ModuleList(self.encoder_networks)

        self.net = []
        hs = [nin] + hidden_sizes + [sum(self.encoded_bins)]

        if hierarchy_embedding_layer_idxs is None:
            hierarchy_embedding_layer_idxs = range(len(hs))
        hierarchy_embedding_layer_idxs = set([idx + 1 for idx in hierarchy_embedding_layer_idxs])

        for i, (h0, h1) in enumerate(zip(hs, hs[1:])):
            first_or_last = i == 0 or i == len(hs) - 2

            if residual_connections and h0 == h1:
                if i in hierarchy_embedding_layer_idxs:
                    self.net.extend(
                        [PartialMaskedResidualBlock(h0, hierarchy_encoding_size, h1,
                                                    activation=activation(inplace=False))])
                else:
                    self.net.extend([MaskedResidualBlock(h0, h1, activation=activation(inplace=False))])

            else:
                if i in hierarchy_embedding_layer_idxs:
                    self.net.extend([PartialMaskedLinear(h0, hierarchy_encoding_size, h1), ])
                else:
                    self.net.extend([MaskedLinear(h0, h1)])

            if not first_or_last:
                self.net.extend([activation(inplace=False)])

        self.net = nn.Sequential(*self.net)

        if self.input_encoding is not None:
            # Input layer should be changed.
            assert self.input_bins is not None
            input_size = 0
            for i, dist_size in enumerate(self.input_bins):
                input_size += self._get_input_encoded_dist_size(i)
            new_layer0 = MaskedLinear(input_size, self.net[0].out_features)
            self.net[0] = new_layer0

        self.update_masks()
        self.orderings = [self.m[-1]]
        self.average_embd = average_embd

    def forward(self, x, hierarchy_batch=None, device=None):
        # input can be dicts here!
        """Calculates unnormalized logits.

        If self.input_bins is not specified, the output units are ordered as:
            [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
        So they can be reshaped as thus and passed to a cross entropy loss:
            out.view(-1, model.nout // model.nin, model.nin)

        Otherwise, they are ordered as:
            [x1, ..., x1], ..., [xn, ..., xn]
        And they can't be reshaped directly.

        Args:
          x: [bs, ncols].
        """

        assert hierarchy_batch is not None
        enc_h_inputs = []
        for h, hb in zip(self.hierarchies, hierarchy_batch):
            # we now have a dict: node_id -> tensor (unencoded)
            # encode all the set evidence
            enc_h_input = {
                node_id: (t[:, 0], self.EncodeInput(t[:, 1:], embedding_idx=h.node_dict[node_id].embedding_idxs)) for
                node_id, t in hb.items()
            }
            if device is not None:
                enc_h_input = {
                    node_id: (s_p.to(device).to(torch.float32), t.to(device).to(torch.float32))
                    for
                    node_id, (s_p, t) in enc_h_input.items()
                }
            enc_h_inputs.append(enc_h_input)

        if device is not None:
            x = x.to(device).to(torch.float32)
        x = self.EncodeInput(x)

        return self.forward_with_encoded_input(x, enc_h_inputs)

    def forward_with_encoded_input(self, enc_input, enc_h_inputs=None):
        assert enc_h_inputs is not None

        # hierarchy bottom-up pass
        h_encodings = []
        for h, hb in zip(self.hierarchies, enc_h_inputs):
            h_enc = h.forward(hb, self.average_embd)
            h_encodings.append(h_enc)

        h_encodings = torch.cat(h_encodings, 1)

        residual = None
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(enc_input)

        for layer in self.net:
            if isinstance(layer, PartialMaskedLinear) or isinstance(layer, PartialMaskedResidualBlock):
                enc_input = layer(enc_input, h_encodings)
            else:
                enc_input = layer(enc_input)

        if residual is not None:
            return enc_input + residual

        return enc_input
