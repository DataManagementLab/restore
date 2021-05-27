import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssar import common
from ssar.hierarchy_made import MADE, MaskedLinear, PartialMaskedLinear, HierarchyMADE, DNNHierarchyNode
from schema_setup.data_preparation.utils import load_pkl, save_pkl

logger = logging.getLogger(__name__)


def run_epoch(split,
              model,
              opt,
              train_data,
              ignore_hierarchy=False,
              hierarchy_tables=None,
              hierarchies=None,
              hierarchy_top_ids=None,
              hierarchy_excluded_ids_per_node=None,
              batches_per_epoch=800,
              val_data=None,
              batch_size=100,
              upto=None,
              epoch_num=None,
              verbose=False,
              log_every=10,
              return_losses=False,
              constant_lr=None,
              warmups=0,
              device='cpu'):
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data.tuples_np if split == 'train' else val_data
    losses = []

    start_t = time.perf_counter()
    shuffled_idx = np.arange(dataset.shape[0])
    np.random.shuffle(shuffled_idx)
    batches_per_epoch = min(batches_per_epoch, int(np.ceil(len(dataset) / batch_size)))
    shuffled_idx = shuffled_idx[:batch_size * batches_per_epoch]

    dataset = dataset[shuffled_idx]
    # this is not aligned with the
    # base_batches = [torch.from_numpy(b) for b in np.array_split(dataset, batches_per_epoch, axis=0)]
    base_batches = [torch.from_numpy(b) for b in np.split(dataset, np.arange(1, batches_per_epoch) * batch_size)]
    logger.info(f"Slicing of {len(base_batches)} AR batches took {time.perf_counter() - start_t:.2f}s")

    # for each hierarchy: sample evidence
    hierarchy_batches = []
    if not ignore_hierarchy:
        for h, table, hids, excluded_ids_per_node in zip(hierarchies, hierarchy_tables, hierarchy_top_ids,
                                                         hierarchy_excluded_ids_per_node):
            # similarly shuffle the excluded ids
            if excluded_ids_per_node is not None:
                excluded_ids_per_node = {id: ex_ids[shuffled_idx] for id, ex_ids in excluded_ids_per_node.items()}
            hierarchy_batches.append(h.sample_batches(table, root_ids=hids[shuffled_idx], batch_size=batch_size,
                                                      excluded_ids_per_node=excluded_ids_per_node))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)

    for step, (xb) in enumerate(base_batches):
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if constant_lr:
                    lr = constant_lr
                elif warmups:
                    t = warmups
                    d_model = model.embed_size
                    global_steps = len(base_batches) * epoch_num + step + 1
                    lr = (d_model ** -0.5) * min(
                        (global_steps ** -.5), global_steps * (t ** -1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break

        xb = xb.to(device).to(torch.float32)
        if not ignore_hierarchy:
            hierarchy_batch = [{
                node_id: all_batches[step].to(device).to(torch.float32) for node_id, all_batches in hb.items()
            } for hb in hierarchy_batches]
        else:
            hierarchy_batch = None

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == 'test' and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, 'update_masks'):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb, hierarchy_batch=hierarchy_batch)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            raise NotImplementedError
            # if mean:
            #     xb = (xb * std) + mean
            # loss = F.binary_cross_entropy_with_logits(
            #     xbhat, xb, size_average=False) / xbhat.size()[0]
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = F.cross_entropy(xbhat, xb.long(), reduction='none') \
                    .sum(-1).mean()
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(
                        model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device))
                    loss = (-logps).mean()

        losses.append(loss.item())

        if step % log_every == 0:
            if split == 'train':
                print(
                    'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}) {:.5f} lr'
                        .format(epoch_num, step, split,
                                loss.item() / np.log(2),
                                loss.item() / np.log(2), lr))
            else:
                print('Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits'.
                      format(epoch_num, step, split, loss.item(),
                             loss.item() / np.log(2)))

        if split == 'train':
            opt.zero_grad()
            loss.backward()
            opt.step()
            # reduce memory consumption by explicitely de-allocating GPU memory
            del xb
            if not ignore_hierarchy:
                del hierarchy_batch

        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))
    if return_losses:
        return losses
    return np.mean(losses)


def report_model(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('Number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


def invert_order(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def make_made(input_encoding, output_encoding, layers, direct_io, scale, residual, cols_to_train, seed,
              inv_order=False, fixed_ordering=None, device='cpu', priors=None):
    if inv_order:
        print('Inverting order!')
        fixed_ordering = invert_order(fixed_ordering)

    model = MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
                     layers if layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=residual,
        fixed_ordering=fixed_ordering,
        column_masking=False,
        priors=priors
    ).to(device)

    return model


def make_hierarchy_made(input_encoding, output_encoding, layers, direct_io, scale, residual, cols_to_train, seed,
                        inv_order=False, fixed_ordering=None, device='cpu', hierarchies=None, hierarchy_tables=None,
                        layers_hierarchy=None, average_embd=True, hierarchy_embedding_layer_idxs=None, priors=None):
    if inv_order:
        print('Inverting order!')
        fixed_ordering = invert_order(fixed_ordering)

    # map column name to id
    column_mapping = {c.name: i for i, c in enumerate(cols_to_train)}

    def relevant_scopes(n, scopes=None):
        if scopes is None:
            scopes = []
        scopes += n.scopes
        for c in n.children:
            relevant_scopes(c, scopes)
        return scopes

    # avoid learning ID columns as well
    rel_scopes = [relevant_scopes(h) for h in hierarchies]
    additional_columns = []

    for table, rel_scope in zip(hierarchy_tables, rel_scopes):
        for i, c in enumerate(table.columns):
            if i not in rel_scope:
                continue
            if c.name in column_mapping.keys():
                continue
            additional_columns.append(c.name)

    # such that column ordering does not change
    additional_columns.sort()
    for c_name in additional_columns:
        column_mapping[c_name] = len(column_mapping)

    # find correct dist sizes for both input_bins (AR model embeddings) and additional_input_dist_sizes (Hierarchy
    # model embeddings)
    input_bins = [c.DistributionSize() for c in cols_to_train]
    additional_input_bins = [0] * (len(column_mapping) - len(cols_to_train))

    for table, rel_scope in zip(hierarchy_tables, rel_scopes):
        for i, c in enumerate(table.columns):
            if i not in rel_scope:
                continue
            col_id = column_mapping[c.name]
            # the distribution sizes should be the same, this is a prior step
            if col_id < len(input_bins):
                assert input_bins[col_id] == c.DistributionSize()
            else:
                add_col_idx = col_id - len(input_bins)
                if additional_input_bins[add_col_idx] > 0:
                    assert additional_input_bins[add_col_idx] == c.DistributionSize()
                additional_input_bins[add_col_idx] = c.DistributionSize()

    # create DNN hierarchy to map to correct col_idx
    def convert_to_dnn_node(n, cols):
        converted_children = [convert_to_dnn_node(c, cols) for c in n.children]
        embedding_idxs = [column_mapping[cols[scope].name] for scope in n.scopes]
        return DNNHierarchyNode(n.id_scope, n.scopes, embedding_idxs, converted_children, n.node_id,
                                n.max_hierarchy_tuples, n.depth)

    dnn_hierarchies = [convert_to_dnn_node(h, ht.columns) for h, ht in zip(hierarchies, hierarchy_tables)]

    model = HierarchyMADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
                     layers if layers > 0 else [512, 256, 512, 128, 1024],
        hierarchy_hidden_sizes=[scale] *
                               layers_hierarchy if layers_hierarchy > 0 else [512, 256],
        hierarchy_embedding_layer_idxs=hierarchy_embedding_layer_idxs,
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        hierarchies=dnn_hierarchies,
        input_bins=[c.DistributionSize() for c in cols_to_train],
        additional_input_bins=additional_input_bins,
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=residual,
        fixed_ordering=fixed_ordering,
        column_masking=False,
        average_embd=average_embd,
        priors=priors
    ).to(device)

    return model


def init_weights(m):
    if type(m) == MaskedLinear or type(m) == nn.Linear or type(m) == PartialMaskedLinear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        if type(m) == PartialMaskedLinear:
            nn.init.xavier_uniform_(m.unmasked_weight)
            nn.init.zeros_(m.unmasked_bias)

    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def train_autoregressive(table, model_dir, model_name, ignore_hierarchy=True, layers_hierarchy=2,
                         input_encoding='binary', output_encoding='one_hot', layers=4, direct_io=True, order=None,
                         residual=True, seed=0, fc_hiddens=128, epochs=20, bs=2048, warmups=8000, device=None,
                         average_embd=True, hierarchy_embedding_layer_idxs=None, hierarchies=None,
                         hierarchy_top_ids=None, hierarchy_excluded_ids_per_node=None,
                         hierarchy_tables=None, batches_per_epoch=800, **kwargs):
    if not ignore_hierarchy:
        assert len(hierarchies) == len(hierarchy_tables) == len(hierarchy_top_ids)

    fixed_ordering = None

    if order is not None:
        logger.info('Using passed-in order:', order)
        fixed_ordering = order

    logger.info(table.data.info())
    table_train = table
    train_data = common.TableDataset(table_train)

    if ignore_hierarchy:
        model = make_made(input_encoding=input_encoding, output_encoding=output_encoding, layers=layers,
                          direct_io=direct_io, scale=fc_hiddens, residual=residual, cols_to_train=table.columns,
                          seed=seed, fixed_ordering=fixed_ordering, device=device, priors=train_data.priors)
    else:
        model = make_hierarchy_made(input_encoding=input_encoding, output_encoding=output_encoding, layers=layers,
                                    direct_io=direct_io, scale=fc_hiddens, residual=residual,
                                    cols_to_train=table.columns, seed=seed, fixed_ordering=fixed_ordering,
                                    device=device, hierarchies=hierarchies, hierarchy_tables=hierarchy_tables,
                                    layers_hierarchy=layers_hierarchy, average_embd=average_embd,
                                    hierarchy_embedding_layer_idxs=hierarchy_embedding_layer_idxs,
                                    priors=train_data.priors)

    report_model(model)
    model_path = os.path.join(model_dir, model_name)
    epoch_path = model_path.replace('.pkl', f'ep_{epochs}.pkl')
    training_time = None
    training_time_path = model_path + '_t.pkl'
    accuracy_path = model_path + '_a.pkl'

    try:
        # load into cpu / gpu:
        model.load_state_dict(torch.load(epoch_path, map_location=torch.device(device)))
        training_time = load_pkl(training_time_path)
        accuracy = load_pkl(accuracy_path)
        model.eval()

    # recompute if weight embeddings do not match
    except (RuntimeError, FileNotFoundError, EOFError) as e:
        # except FileNotFoundError as e:
        logger.info('Applying InitWeight()')
        model.apply(init_weights)

        opt = torch.optim.Adam(list(model.parameters()), 2e-4)

        bs = bs
        log_every = 200

        hierarchy_tables_np = None
        if not ignore_hierarchy:
            hierarchy_tables_np = [common.TableDataset(ht).tuples_np for ht in hierarchy_tables]

        train_losses = []
        train_start = time.perf_counter()
        for epoch in range(epochs):
            mean_epoch_train_loss = run_epoch('train',
                                              model,
                                              opt,
                                              batches_per_epoch=batches_per_epoch,
                                              ignore_hierarchy=ignore_hierarchy,
                                              train_data=train_data,
                                              hierarchy_tables=hierarchy_tables_np,
                                              hierarchies=hierarchies,
                                              hierarchy_top_ids=hierarchy_top_ids,
                                              hierarchy_excluded_ids_per_node=hierarchy_excluded_ids_per_node,
                                              val_data=train_data,
                                              batch_size=bs,
                                              epoch_num=epoch,
                                              log_every=log_every,
                                              warmups=warmups,
                                              device=device)
            training_time = time.perf_counter() - train_start
            if epoch % 1 == 0:
                logger.info('epoch {} train loss {:.4f} nats / {:.4f} bits'.format(
                    epoch, mean_epoch_train_loss,
                    mean_epoch_train_loss / np.log(2)))
                logger.info('time since start: {:.1f} secs'.format(training_time))

            train_losses.append(mean_epoch_train_loss)

        accuracy = train_losses[-1]
        save_pkl(training_time_path, training_time)
        save_pkl(accuracy_path, accuracy)
        torch.save(model.state_dict(), epoch_path)
        logger.info(f"Saved model to {epoch_path}")

    return accuracy, training_time, model
