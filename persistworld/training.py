"""Training on streams (Sec. 3.6): segments from the initial state, truncated backpropagation.

A segment runs up to eight consecutive capture groups of one scene from the initial state; losses
are placed at keyframes whose state depths are drawn from 1..8 (plan.py). Gradients are truncated
to the last ``k`` writes before each loss (k = 2): a no-gradient pass advances the state, and for a
loss at depth d the state after write d-k is restored and the last k writes are redone with
gradient, then the reader forecasts and the loss is backpropagated at once. Memory does not grow
with the segment length. Every image is encoded once: groups inside some loss window are encoded
with gradient, their features enter the writes as detached leaves, and the accumulated feature
gradient is pushed into the encoder once.
"""
import math

import torch

from . import streaming as st
from .losses import frame_loss


def segment_backward(model, frames, sups, depths, k, count, sums, device, retention):
    """Forward and backward of one segment.

    ``frames``: CPU inputs of depths 1..D (D = max(depths)); ``sups``: {depth: CPU supervision}. Each
    loss is divided by ``count`` (the loss points of the whole update) and backpropagated through its
    last k writes. Accumulates sums += (loss, ray term, losses, writes).
    """
    writer = model.writer
    bins = getattr(writer, 'depth_bins', 0)
    depths = sorted(depths)
    D = depths[-1]
    if len(frames) != D:
        raise ValueError('One frame per state depth up to the deepest loss')
    need = {j for d in depths for j in range(max(1, d-k+1), d+1)}   # writes that carry gradient
    gpu = [st.transfer(c, device) for c in frames]
    main = st.Stream(retention)
    snaps = {}
    enc = {}
    leaf = {}

    def flush(limit):
        """Push the accumulated feature gradients of groups no later loss needs into the encoder."""
        for j in [j for j in enc if j < limit]:
            g = leaf[j]['features'].grad
            if g is not None:
                enc[j]['features'].backward(g)
            del enc[j], leaf[j]
    writer.record_depth = False
    for d in range(1, D+1):
        b = gpu[d-1]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if d in need:
                e = model.encoder.encode(b['rgb'])
                enc[d] = e
                leaf[d] = dict(e, features=e['features'].detach().requires_grad_(e['features'].requires_grad))
                encoded = dict(e, features=e['features'].detach())
            else:
                with torch.no_grad():
                    encoded = model.encoder.encode(b['rgb'])
            with torch.no_grad():
                main.observe(model, b, encoded=encoded, read=False)
                main.finish()
            snaps[d] = main.snapshot()
        snaps.pop(d-k-1, None)
        if d not in depths:
            continue
        branch = st.Stream(retention)
        if d-k > 0:
            branch.restore(snaps[d-k])
        for j in range(max(1, d-k+1), d+1):
            last = j == d
            writer.record_depth = bool(bins) and last
            writer.depth_records.clear()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                cur = branch.observe(model, gpu[j-1], encoded=leaf[j], read=last)
                fut = branch.forecast(model, gpu[j-1]) if last else None
            branch.finish()
        supd = {key: v.to(device, non_blocking=v.is_pinned()) for key, v in sups[d].items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, ray = frame_loss(model, b, supd, fut, cur, lambda: writer.depth_records[-1][0, 0])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        (loss/count).backward()
        writer.record_depth = False
        ray = loss.detach()*0 if ray is None else ray
        sums += torch.stack([loss.detach().float(), ray.float(), torch.ones((), device=device),
                             torch.zeros((), device=device)])
        del branch, cur, fut, loss, supd
        later = [x for x in depths if x > d]
        flush(later[0]-k+1 if later else D+1)
    flush(D+1)
    sums[3] += D


def learning_rate(lr, warmup, global_step, per_epoch, epochs, decay_from, floor):
    """Linear warmup over ``warmup`` updates, constant, then a cosine decay to lr*floor over the epochs
    from ``decay_from`` to ``epochs``."""
    t = (global_step+1-decay_from*per_epoch)/max(1, (epochs-decay_from)*per_epoch)
    decay = 1. if t <= 0 else floor+(1-floor)*.5*(1+math.cos(math.pi*min(1., t)))
    return lr*min(1., (global_step+1)/warmup)*decay


def pretrained_backbone(name):
    """Parameters trained at the backbone learning-rate scale (the tuned DINOv3 blocks)."""
    return name.startswith('encoder.')


def make_optimizer(model, lr, backbone_scale):
    """AdamW over the trainable parameters (the tuned encoder blocks at ``backbone_scale``) and their list.

    Each parameter group carries its learning-rate ``scale``; the trainers set lr = base lr * scale.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    backbone = [p for n, p in named if pretrained_backbone(n)]
    rest = [p for n, p in named if not pretrained_backbone(n)]
    groups = [dict(params=rest, scale=1.)]+([dict(params=backbone, scale=backbone_scale)] if backbone else [])
    return torch.optim.AdamW(groups, lr=lr, weight_decay=0.), [p for _, p in named]


def all_reduce_gradients(params, sums=None):
    """Sum the gradients of all ranks (one flat all-reduce per dtype)."""
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    for dtype in {p.grad.dtype for p in params}:
        grads = [p.grad for p in params if p.grad.dtype == dtype]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        for g, f in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(f)
    if sums is not None:
        dist.all_reduce(sums)


INPUT_KEYS = ('rgb', 'world_from_ego', 'image_from_world', 'times', 'future_world_from_ego', 'future_times',
              'current_world_from_ego')


def window_inputs(batch, device):
    """Model inputs of a windowed query on ``device`` (labels and masks never enter the model)."""
    from .model import host_sync as hs
    out = {k: batch[k] for k in ('scene', 'token', 'anchor_index', 'frame_ids', 'history_mode')}
    out.update({k: hs.to_device(batch[k], device) for k in INPUT_KEYS})
    return out


def current_ray_logits(writer, batch, groups_per_write):
    """Ray-head logits [N,P,D] of the current capture group from the writer's records of a windowed query."""
    from .model.persistworld import window_blocks
    current = tuple(batch['frame_ids'][-1])
    pieces = window_blocks(batch, groups_per_write)
    if len(pieces) != len(writer.depth_records):
        raise RuntimeError('Ray records do not match the writes')
    for piece, record in zip(pieces, writer.depth_records):
        for pos, ids in enumerate(piece['frame_ids']):
            if tuple(ids) == current:
                return record[0, pos]
    raise RuntimeError('Current capture group not found')


def window_loss(model, batch, sup, device):
    """Loss of one windowed query: a fresh state from the eight-group input window, then forecast."""
    writer = model.writer
    bins = getattr(writer, 'depth_bins', 0)
    writer.record_depth = bool(bins)
    writer.depth_records.clear()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        future, current = model.forward_query(window_inputs(batch, device), current_readout=True)
        sup = {k: v.to(device, non_blocking=v.is_pinned()) for k, v in sup.items()}
        loss, ray = frame_loss(model, batch, sup, future, current,
                               lambda: current_ray_logits(writer, batch, model.config.groups_per_write))
    writer.depth_records.clear()
    return loss, ray
