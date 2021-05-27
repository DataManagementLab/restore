"""MADE and ResMADE."""
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from join_completion.query_compilation.planning import PredictabilityScore
from ssar.masked_blocks import MaskedLinear, MaskedResidualBlock

logger = logging.getLogger(__name__)


class MADE(nn.Module):

    def __init__(
            self,
            nin,
            hidden_sizes,
            nout,
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
            skip_net=False,
            max_embedding_size=8192,
            priors=None
    ):
        """MADE.

        Args:
          nin: integer; number of input variables.  Each input variable
            represents a column.
          hidden sizes: a list of integers; number of units in hidden layers.
          nout: integer; number of outputs, the sum of all input variables'
            domain sizes.
          num_masks: number of orderings + connectivity masks to cycle through.
          natural_ordering: force natural ordering of dimensions, don't use
            random permutations.
          input_bins: classes each input var can take on, e.g., [5, 2] means
            input x1 has values in {0, ..., 4} and x2 in {0, 1}.  In other
            words, the domain sizes.
          activation: the activation to use.
          do_direct_io_connections: whether to add a connection from inputs to
            output layer.  Helpful for information flow.
          input_encoding: input encoding mode, see EncodeInput().
          output_encoding: output logits decoding mode, either 'embed' or
            'one_hot'.  See logits_for_col().
          embed_size: int, embedding dim.
          residual_connections: use ResMADE?  Could lead to faster learning.
          column_masking: if True, turn on column masking during training time,
            which enables the wildcard skipping optimization during inference.
            Recommended to be set for any non-trivial datasets.
          seed: seed for generating random connectivity masks.
          fixed_ordering: variable ordering to use.  If specified, order[i]
            maps natural index i -> position in ordering.  E.g., if order[0] =
            2, variable 0 is placed at position 2.
        """
        super().__init__()
        print('fixed_ordering', fixed_ordering, 'seed', seed,
              'natural_ordering', natural_ordering)
        self.nin = nin
        assert input_encoding in ['embed']
        self.input_encoding = input_encoding
        assert output_encoding in ['embed']
        self.embed_size = self.emb_dim = embed_size
        self.output_encoding = output_encoding
        self.activation = activation
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        self.input_bins = input_bins
        self.extended_input_bins = self.input_bins
        if additional_input_bins is not None:
            self.extended_input_bins = self.input_bins + additional_input_bins
        self.do_direct_io_connections = do_direct_io_connections
        self.column_masking = column_masking
        self.residual_connections = residual_connections
        self.max_embedding_size = max_embedding_size
        self.column_factorized = False
        self.priors = priors

        self.fixed_ordering = fixed_ordering
        if fixed_ordering is not None:
            assert num_masks == 1
            print('** Fixed ordering {} supplied, ignoring natural_ordering'.
                  format(fixed_ordering))

        assert self.input_bins

        self.no_subcolumns = [
            math.ceil(math.log(dist_size, self.max_embedding_size)) if dist_size > max_embedding_size else 0 for
            dist_size in self.extended_input_bins]
        # for the last subcolumn we might not need all self.max_embedding_size positions
        self.last_factor_size = [
            math.ceil(self.extended_input_bins[i] / (self.max_embedding_size ** (self.no_subcolumns[i] - 1))) if
            self.no_subcolumns[i] > 0 else 0 for i in range(len(self.extended_input_bins))
        ]
        self.input_encoded_dist_sizes = []
        for col_idx in range(len(self.extended_input_bins)):
            enc_dist_size = self.extended_input_bins[col_idx]
            if enc_dist_size > self.max_embedding_size:
                enc_dist_size = self.no_subcolumns[col_idx] * self.embed_size
            else:
                enc_dist_size = min(enc_dist_size, self.embed_size)
            self.input_encoded_dist_sizes.append(enc_dist_size)

        encoded_bins = list(
            map(self._get_output_encoded_dist_size, range(len(self.input_bins))))
        self.input_bins_encoded = list(
            map(self._get_input_encoded_dist_size, range(len(self.input_bins))))
        self.input_bins_encoded_cumsum = np.cumsum(
            list(map(self._get_input_encoded_dist_size, range(len(self.input_bins)))))
        print('encoded_bins (output)', encoded_bins)
        self.encoded_bins = encoded_bins
        print('encoded_bins (input)', self.input_bins_encoded)

        if not skip_net:
            hs = [nin] + hidden_sizes + [sum(encoded_bins)]
            self.net = []
            for h0, h1 in zip(hs, hs[1:]):
                if residual_connections:
                    if h0 == h1:
                        self.net.extend([
                            MaskedResidualBlock(
                                h0, h1, activation=activation(inplace=False))
                        ])
                    else:
                        self.net.extend([
                            MaskedLinear(h0, h1),
                        ])
                else:
                    self.net.extend([
                        MaskedLinear(h0, h1),
                        activation(inplace=True),
                    ])
            if not residual_connections:
                self.net.pop()
            self.net = nn.Sequential(*self.net)

            if self.input_encoding is not None:
                # Input layer should be changed.
                assert self.input_bins is not None
                input_size = 0
                for i, dist_size in enumerate(self.input_bins):
                    input_size += self._get_input_encoded_dist_size(i)
                new_layer0 = MaskedLinear(input_size, self.net[0].out_features)
                self.net[0] = new_layer0

        if self.output_encoding == 'embed':
            assert self.input_encoding == 'embed'

        if self.input_encoding == 'embed':
            self.embeddings = nn.ModuleList()
            for i, dist_size in enumerate(self.extended_input_bins):
                if dist_size <= self.embed_size:
                    embed = None
                # column factorization
                elif dist_size >= self.max_embedding_size:
                    self.column_factorized = True
                    embed = nn.ModuleList()
                    for sub_idx in range(self.no_subcolumns[i]):
                        # last factor
                        if sub_idx == self.no_subcolumns[i] - 1:
                            embed.append(nn.Embedding(self.last_factor_size[i], self.embed_size))
                        else:
                            embed.append(nn.Embedding(self.max_embedding_size, self.embed_size))
                else:
                    embed = nn.Embedding(dist_size, self.embed_size)
                self.embeddings.append(embed)

        # Learnable [MASK] transform_tensors.
        if self.column_masking:
            self.unk_embeddings = nn.ParameterList()
            for i, dist_size in enumerate(self.input_bins):
                self.unk_embeddings.append(
                    nn.Parameter(torch.zeros(1, self.input_bins_encoded[i])))

        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = seed if seed is not None else 11123
        self.init_seed = self.seed

        self.direct_io_layer = None
        self.logit_indices = np.cumsum(encoded_bins)
        self.m = {}

        if not skip_net:
            self.update_masks()
            self.orderings = [self.m[-1]]

        # Optimization: cache some values needed in EncodeInput().
        self.bin_as_onehot_shifts = None

    def _build_or_update_direct_io(self):
        assert self.nout > self.nin and self.input_bins is not None
        direct_nin = self.net[0].in_features
        direct_nout = self.net[-1].out_features
        if self.direct_io_layer is None:
            self.direct_io_layer = MaskedLinear(direct_nin, direct_nout)
        mask = np.zeros((direct_nout, direct_nin), dtype=np.uint8)

        if self.natural_ordering:
            curr = 0
            for i in range(self.nin):
                dist_size = self._get_input_encoded_dist_size(i)
                # Input i connects to groups > i.
                mask[self.logit_indices[i]:, curr:dist_size] = 1
                curr += dist_size
        else:
            # Inverse: ord_idx -> natural idx.
            inv_ordering = [None] * self.nin
            for natural_idx in range(self.nin):
                inv_ordering[self.m[-1][natural_idx]] = natural_idx

            for ord_i in range(self.nin):
                nat_i = inv_ordering[ord_i]
                # x_(nat_i) in the input occupies range [inp_l, inp_r).
                inp_l = 0 if nat_i == 0 else self.input_bins_encoded_cumsum[
                    nat_i - 1]
                inp_r = self.input_bins_encoded_cumsum[nat_i]
                assert inp_l < inp_r

                for ord_j in range(ord_i + 1, self.nin):
                    nat_j = inv_ordering[ord_j]
                    # Output x_(nat_j) should connect to input x_(nat_i); it
                    # occupies range [out_l, out_r) in the output.
                    out_l = 0 if nat_j == 0 else self.logit_indices[nat_j - 1]
                    out_r = self.logit_indices[nat_j]
                    assert out_l < out_r
                    mask[out_l:out_r, inp_l:inp_r] = 1
        mask = mask.T
        self.direct_io_layer.set_mask(mask)

    def _get_input_encoded_dist_size(self, col_idx):
        # column factorization
        return self.input_encoded_dist_sizes[col_idx]

    def _get_output_encoded_dist_size(self, col_idx):
        return self._get_input_encoded_dist_size(col_idx)

    def update_masks(self, invoke_order=None):
        """Update m() for all layers and change masks correspondingly.

        No-op if "self.num_masks" is 1.
        """
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_sizes)

        ### Precedence of several params determining ordering:
        #
        # invoke_order
        # orderings
        # fixed_ordering
        # natural_ordering
        #
        # from high precedence to low.

        if invoke_order is not None:
            found = False
            for i in range(len(self.orderings)):
                if np.array_equal(self.orderings[i], invoke_order):
                    found = True
                    break
            assert found, 'specified={}, avail={}'.format(
                ordering, self.orderings)
            # orderings = [ o0, o1, o2, ... ]
            # seeds = [ init_seed, init_seed+1, init_seed+2, ... ]
            rng = np.random.RandomState(self.init_seed + i)
            self.seed = (self.init_seed + i + 1) % self.num_masks
            self.m[-1] = invoke_order
        elif hasattr(self, 'orderings'):
            # Cycle through the special orderings.
            rng = np.random.RandomState(self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = self.orderings[self.seed % 4]
        else:
            rng = np.random.RandomState(self.seed)
            self.seed = (self.seed + 1) % self.num_masks
            self.m[-1] = np.arange(
                self.nin) if self.natural_ordering else rng.permutation(
                self.nin)
            if self.fixed_ordering is not None:
                self.m[-1] = np.asarray(self.fixed_ordering)

        if self.nin > 1:
            for l in range(L):
                if self.residual_connections:
                    # Sequential assignment for ResMade: https://arxiv.org/pdf/1904.05626.pdf
                    self.m[l] = np.array([(k - 1) % (self.nin - 1)
                                          for k in range(self.hidden_sizes[l])])
                else:
                    # Samples from [0, ncols - 1).
                    self.m[l] = rng.randint(self.m[l - 1].min(),
                                            self.nin - 1,
                                            size=self.hidden_sizes[l])
        else:
            # This should result in first layer's masks == 0.
            # So output units are disconnected to any inputs.
            for l in range(L):
                self.m[l] = np.asarray([-1] * self.hidden_sizes[l])

        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        if self.nout > self.nin:
            # Last layer's mask needs to be changed.

            if self.input_bins is None:
                k = int(self.nout / self.nin)
                # Replicate the mask across the other outputs
                # so [x1, x2, ..., xn], ..., [x1, x2, ..., xn].
                masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
            else:
                # [x1, ..., x1], ..., [xn, ..., xn] where the i-th list has
                # input_bins[i - 1] many elements (multiplicity, # of classes).
                mask = np.asarray([])
                for k in range(masks[-1].shape[0]):
                    tmp_mask = []
                    for idx, x in enumerate(zip(masks[-1][k], self.input_bins)):
                        mval, nbins = x[0], self._get_output_encoded_dist_size(idx)
                        tmp_mask.extend([mval] * nbins)
                    tmp_mask = np.asarray(tmp_mask)
                    if k == 0:
                        mask = tmp_mask
                    else:
                        mask = np.vstack([mask, tmp_mask])
                masks[-1] = mask

        if self.input_encoding is not None:
            # Input layer's mask should be changed.

            assert self.input_bins is not None
            # [nin, hidden].
            mask0 = masks[0]
            new_mask0 = []
            for i, dist_size in enumerate(self.input_bins):
                dist_size = self._get_input_encoded_dist_size(i)
                # [dist size, hidden]
                new_mask0.append(
                    np.concatenate([mask0[i].reshape(1, -1)] * dist_size,
                                   axis=0))
            # [sum(dist size), hidden]
            new_mask0 = np.vstack(new_mask0)
            masks[0] = new_mask0

        layers = [
            l for l in self.net if isinstance(l, MaskedLinear) or
                                   isinstance(l, MaskedResidualBlock)
        ]
        assert len(layers) == len(masks), (len(layers), len(masks))
        for l, m in zip(layers, masks):
            l.set_mask(m)

        if self.do_direct_io_connections:
            self._build_or_update_direct_io()

    def Embed(self, data, embedding_idx=None, natural_col=None, out=None):
        if data is None:
            if out is None:
                return self.unk_embeddings[natural_col]
            out.copy_(self.unk_embeddings[natural_col])
            return out

        bs = data.size()[0]
        y_embed = []
        data = data.long()

        if embedding_idx is None:
            # col_idxs = range(len(self.input_bins))
            embedding_idx = range(len(self.input_bins))

        for i, col_idx in enumerate(embedding_idx):
            assert len(embedding_idx) == data.shape[1]
            # Wildcard column? use -1 as special token.
            # Inference pass only (see estimators.py).
            skip = data[0][i] < 0
            coli_dom_size = self.extended_input_bins[col_idx]

            # Embed?
            if coli_dom_size > self.embed_size:
                # Column factorized
                if coli_dom_size >= self.max_embedding_size:
                    remaining_data = data[:, i]
                    for i in range(self.no_subcolumns[col_idx]):
                        factor_i = torch.remainder(remaining_data, self.max_embedding_size)
                        embed_i = self.embeddings[col_idx][i](factor_i)
                        y_embed.append(embed_i)
                        remaining_data = torch.floor_divide(remaining_data, self.max_embedding_size)
                    if self.column_masking:
                        raise NotImplementedError

                else:
                    col_i_embs = self.embeddings[col_idx](data[:, i])
                    if not self.column_masking:
                        y_embed.append(col_i_embs)
                    else:
                        dropped_repr = self.unk_embeddings[col_idx]

                        def dropout_p():
                            return np.random.randint(0, self.nin) / self.nin

                        # During training, non-dropped 1's are scaled by
                        # 1/(1-p), so we clamp back to 1.
                        batch_mask = torch.clamp(
                            torch.dropout(torch.ones(bs, 1, device=data.device),
                                          p=dropout_p(),
                                          train=self.training), 0, 1)
                        y_embed.append(batch_mask * col_i_embs +
                                       (1. - batch_mask) * dropped_repr)
            else:
                if skip:
                    y_embed.append(self.unk_embeddings[col_idx])
                    continue
                y_onehot = torch.zeros(bs,
                                       coli_dom_size,
                                       device=data.device)
                y_onehot.scatter_(1, data[:, i].view(-1, 1), 1)
                if self.column_masking:

                    def dropout_p():
                        return np.random.randint(0, self.nin) / self.nin

                    # During training, non-dropped 1's are scaled by
                    # 1/(1-p), so we clamp back to 1.
                    batch_mask = torch.clamp(
                        torch.dropout(torch.ones(bs, 1, device=data.device),
                                      p=dropout_p(),
                                      train=self.training), 0, 1)
                    y_embed.append(batch_mask * y_onehot +
                                   (1. - batch_mask) *
                                   self.unk_embeddings[col_idx])
                else:
                    y_embed.append(y_onehot)
        return torch.cat(y_embed, 1)

    def EncodeInput(self, data, embedding_idx=None, natural_col=None, out=None):
        """"Warning: this could take up a significant portion of a forward pass.

        Args:
          natural_col: if specified, 'data' has shape [N, 1] corresponding to
              col-'natural-col'.  Otherwise 'data' corresponds to all cols.
          out: if specified, assign results into this Tensor storage.
        """
        return self.Embed(data, embedding_idx=embedding_idx, natural_col=natural_col, out=out)

    def forward(self, x, hierarchy_batch=None, device=None):
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
        assert hierarchy_batch is None

        if device is not None:
            x = x.to(device).to(torch.float32)
        x = self.EncodeInput(x)

        return self.forward_with_encoded_input(x)

    def forward_with_encoded_input(self, x, enc_h_inputs=None):
        assert enc_h_inputs is None
        if self.direct_io_layer is not None:
            residual = self.direct_io_layer(x)
            return self.net(x) + residual

        return self.net(x)

    def logits_for_col(self, idx, logits):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        assert self.input_bins is not None

        if idx == 0:
            logits_for_var = logits[:, :self.logit_indices[0]]
        else:
            logits_for_var = logits[:, self.logit_indices[idx - 1]:self.logit_indices[idx]]
        if self.output_encoding != 'embed':
            return logits_for_var

        embed = self.embeddings[idx]

        if embed is None:
            # Can be None for small-domain columns.
            return logits_for_var

        # Otherwise, dot with embedding matrix to get the true logits.
        # [bs, emb] * [emb, dom size for idx]
        return torch.matmul(logits_for_var, embed.weight.t())

    def logits_for_subcol(self, idx, sub_idx, logits):
        """Returns the logits (vector) corresponding to log p(x_i | x_(<i)).

        Args:
          idx: int, in natural (table) ordering.
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.

        Returns:
          logits_for_col: [batch size, domain size for column idx].
        """
        assert self.input_bins is not None

        start_logit_idx = self.emb_dim * sub_idx
        if idx > 0:
            start_logit_idx += self.logit_indices[idx - 1]
        end_logit_idx = start_logit_idx + self.emb_dim

        logits_for_var = logits[:, start_logit_idx:end_logit_idx]

        embed = self.embeddings[idx][sub_idx]

        # Otherwise, dot with embedding matrix to get the true logits.
        # [bs, emb] * [emb, dom size for idx]
        return torch.matmul(logits_for_var, embed.weight.t())

    def nll(self, logits, data):
        """Calculates -log p(data), given logits (the conditionals).

        Args:
          logits: [batch size, hidden] where hidden can either be sum(dom
            sizes), or emb_dims.
          data: [batch size, nin].

        Returns:
          nll: [batch size].
        """
        if data.dtype != torch.long:
            data = data.long()
        nll = torch.zeros(logits.size()[0], device=logits.device)
        for i in range(self.nin):
            if self.no_subcolumns[i] > 0:
                remaining_data = data[:, i]
                for sub_idx in range(self.no_subcolumns[i]):
                    factor_i = torch.remainder(remaining_data, self.max_embedding_size)
                    logits_i = self.logits_for_subcol(i, sub_idx, logits)
                    nll += F.cross_entropy(logits_i, factor_i, reduction='none')
                    remaining_data = torch.floor_divide(remaining_data, self.max_embedding_size)
            else:
                logits_i = self.logits_for_col(i, logits)
                nll += F.cross_entropy(logits_i, data[:, i], reduction='none')

        return nll

    def sample(self, no_samples=1, evidence=None, hierarchy_batches=None, device=None, end_idx=None,
               pos_weights=None, bs=10000, suppress_nan=True, pos_weights_idx=None, percentile_idxs=None,
               percentile_inverted=None, percentile=None, percentile_idxs_values=None, predictability_score=None):
        assert self.natural_ordering
        assert self.input_bins and self.nout > self.nin

        sampled = np.zeros((no_samples, self.nin))
        start_idx = 0

        if evidence is not None:
            sampled = np.zeros((evidence.shape[0], self.nin))
            start_idx = evidence.shape[1]
            sampled[:, np.arange(0, start_idx)] = evidence

        no_batches = math.ceil(sampled.shape[0] / bs)

        if hierarchy_batches is not None:
            hierarchy_batches = [h.split_batches(bs, no_batches, hbs)
                                 for hbs, h in zip(hierarchy_batches, self.hierarchies)]

        if end_idx is None:
            end_idx = self.nin - 1

        # we are done then
        if end_idx < start_idx:
            return sampled

        with torch.no_grad():

            if pos_weights is not None:
                pos_weights = np.array(pos_weights)
                # do not normalize to avoid problems
                # pos_weights /= np.sum(pos_weights)
                pos_weights = torch.from_numpy(pos_weights).to(device).to(torch.float32)

            if percentile is not None:
                # priors as gpu
                prior_dict = dict()
                for idx, vals, perc_inv in zip(percentile_idxs, percentile_idxs_values, percentile_inverted):
                    prior_dict[idx] = (torch.from_numpy(self.priors[idx]).to(device).to(torch.float32), vals, perc_inv)

            for batch_i in range(no_batches):
                batch_start = batch_i * bs
                batch_end = min((batch_i + 1) * bs, sampled.shape[0])

                hierarchy_batch = None
                if hierarchy_batches is not None:
                    hierarchy_batch = [
                        {node_id: torch.from_numpy(ev[batch_i]).to(device).to(torch.float32) for node_id, ev in
                         hbs.items()} for hbs in hierarchy_batches]

                # already send to correct device
                batch = torch.from_numpy(sampled[batch_start:batch_end, :]).to(device).to(torch.float32)
                if batch_i > 0 and batch_i % 1 == 0:
                    logger.info(f"\t\tSampling batch {batch_i}/{no_batches} (batch size: {bs})")

                for i in range(start_idx, end_idx + 1):
                    if self.no_subcolumns[i] > 0:
                        # not yet implemented
                        assert i not in percentile_idxs
                        assert pos_weights is None
                        base = 1

                        for sub_idx in range(self.no_subcolumns[i]):
                            logits = self.forward(batch, hierarchy_batch=hierarchy_batch, device=device)
                            probs = torch.softmax(self.logits_for_subcol(i, sub_idx, logits), -1)
                            s = torch.multinomial(probs, 1)
                            s *= base
                            batch[:, i] += s.view(-1, )
                            base *= self.max_embedding_size

                    else:
                        logits = self.forward(batch, hierarchy_batch=hierarchy_batch, device=device)
                        probs = torch.softmax(self.logits_for_col(i, logits), -1)

                        if percentile is not None and i in percentile_idxs:
                            prior, val, perc_inv = prior_dict[i]
                            # continous case

                            current_percentile = percentile
                            if val == ['none']:
                                # take highest value of continous attribute
                                if percentile > 50:
                                    val = [probs.shape[1] - 1]
                                # take lowest value of cont. attribute very often
                                else:
                                    current_percentile = 100 - percentile
                                    val = [1]
                                # raise NotImplementedError("Not yet implemented for continous. Idea: simple take
                                #   min/max as value.")

                            if len(val) > 1:
                                raise NotImplementedError("Not yet implemented for bias on several values")
                            val = int(val[0])

                            # compute score (how predictable is the attribute according to the model)
                            # in our case: either approximated by how much we differ from prior for value of query
                            # or the KL divergence
                            if predictability_score == PredictabilityScore.KL_DIV_PRIOR:
                                kl_div = (probs * (probs / prior).log()).sum(dim=1)
                                # normalize
                                scores = 1 - torch.exp(-kl_div)
                            elif predictability_score == PredictabilityScore.PRIOR_VAL:
                                scores_pos = (probs[:, val] - prior[val]) / (1 - prior[val])
                                scores_neg = (prior[val] - probs[:, val]) / (prior[val])
                                scores = torch.clamp(scores_pos, min=0, max=1) + torch.clamp(scores_neg, min=0, max=1)
                            else:
                                raise NotImplementedError
                            scores = torch.clamp(scores, min=0, max=1)
                            scores[torch.isnan(scores)] = 0

                            # weighted sum with val
                            p_biased = torch.ones_like(probs)

                            # how often the value occurs in the biased distribution
                            p_biased_prob = current_percentile / 100
                            if perc_inv:
                                p_biased_prob = 1 - current_percentile

                            # scale this based on prior, a 5 percentile of 5% does not make sense if the attribute
                            # only occurs in 1% in the prior
                            if p_biased_prob > 0.5:
                                p_biased_prob = prior[val] + (p_biased_prob - 0.5) / 0.5 * (1 - prior[val])
                            else:
                                p_biased_prob = prior[val] - (0.5 - p_biased_prob) / 0.5 * prior[val]

                            # assert np.all(p_biased.cpu().numpy() >= 0)
                            p_biased *= (1 - p_biased_prob) / (p_biased.shape[1] - 1)
                            # assert np.all(p_biased.cpu().numpy() >= 0)
                            # p_biased[:] = (1 - p_biased_prob) / (1 - prior[val]) * prior

                            # assert np.all(scores.cpu().numpy() <= 1)
                            # assert np.all(scores.cpu().numpy() >= 0)
                            p_biased[:, val] = p_biased_prob
                            # assert np.all(p_biased.cpu().numpy() >= 0)

                            scores = scores.view(-1, 1)
                            probs = probs * scores + p_biased * (1 - scores)
                            # assert np.all(probs.cpu().numpy() >= 0)

                        if pos_weights is not None and (pos_weights_idx is None or pos_weights_idx == i):
                            probs *= pos_weights
                            # to prevent that weights are zero
                            probs[~(torch.sum(probs, dim=1) > 0)] = pos_weights
                        elif suppress_nan:
                            probs[:, 0] = 0

                        assert np.all(probs.cpu().numpy() >= 0)
                        s = torch.multinomial(probs, 1)
                        batch[:, i] = s.view(-1, )

                sampled[batch_start:batch_end, :] = batch.cpu().numpy()
                # improved memory management
                if hierarchy_batches is not None:
                    del hierarchy_batch
                del batch

        return sampled
