"""Unified occupancy objective (Sec. 3.5 of the paper).

Every voxel is supervised with a focal score on the softmax probabilities p_c of its class
distribution, applied to every class as a binary event. With f(q) = -(1-q)^2 log q, a voxel of
label y receives

    l(p, y) = w_y [ a+_y f(p_y) + sum_{c != y} a-_c f(1 - p_c) ],

with a+_c ~ pi_c^(-1/2), a-_c ~ (1 - pi_c)^(-1/2) from the training class priors pi (shared
normalization), w_free = 4 and 1 otherwise. The grid term averages l over the voxels inside the
target mask. The ray view (the writer's ray-depth head, rendered per section from the current
ground truth) gets the same objective on its two classes, averaged over its valid sections. The
loss at one frame is

    L = mean_k s L_grid(t_k) + 0.25 (s L_grid(t_0) + L_ray),

over the supervised future horizons k (1, 2, 3 s and one of 4..8 s). The constant s = 2.2949
(``loss_scale`` in the config) was set once from the gradient scale of the grid term on a fixed
calibration batch and is kept for all models; it does not multiply the ray term.
"""
import math

import torch

from .model import ray_depth


def validate_free_weight(weight, free_id, classes):
    """Check the weight of voxels labelled free and the free class index."""
    if (isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight)
            or not 0 < weight <= 1e6):
        raise ValueError('Expected positive finite GT-free example weight')
    if type(free_id) is not int or not 0 <= free_id < classes:
        raise ValueError('Invalid free class index')


def objective_spec(class_priors, free_weight=4., loss_scale=1., current_weight=.25, num_classes=18, free_id=17,
                   prior_exponent=.5):
    """Class weights a+ and a- of the focal objective from the training class priors."""
    priors = [float(p) for p in class_priors]
    if len(priors) != num_classes or any(not math.isfinite(p) or p <= 0 for p in priors):
        raise ValueError('Every class needs a positive finite prior')
    if not math.isclose(sum(priors), 1., rel_tol=0, abs_tol=1e-10):
        raise ValueError('Class priors must sum to one')
    if any(p >= 1 for p in priors):
        raise ValueError('Every class and its complement require positive support')
    validate_free_weight(free_weight, free_id, num_classes)
    exponent = float(prior_exponent)
    positive = [p ** -exponent for p in priors]
    negative = [(1 - p) ** -exponent for p in priors]
    normalizer = sum(p * a + (1 - p) * b for p, a, b in zip(priors, positive, negative))
    return dict(field_priors=priors, positive_weights=[v / normalizer for v in positive],
                negative_weights=[v / normalizer for v in negative], train_loss_scale=float(loss_scale),
                free_target_weight=float(free_weight), free_id=free_id, num_classes=num_classes,
                current_weight=float(current_weight))


def stable_log_probabilities(x):
    """Stable log(p) and log(1-p), including confidently wrong logits."""
    x = x - x.max(-1, keepdim=True).values
    logden = x.logsumexp(-1, keepdim=True)
    logp = x - logden
    winner = torch.zeros_like(x, dtype=torch.bool).scatter_(1, x.argmax(-1, keepdim=True), True)
    # Non-winning probabilities are <= 1/2; log1p(-1) on the unused winner branch would give 0*inf NaNs.
    safe_p = logp.exp().masked_fill(winner, 0.)
    ordinary = torch.log1p(-safe_p)
    winning = x.masked_fill(winner, -torch.inf).logsumexp(-1, keepdim=True) - logden
    return logp, torch.where(winner, winning, ordinary)


def focal_field_loss(logits, labels, mask, positive_weights, negative_weights, scale, free_target_weight, free_id):
    """Mean focal loss over the valid voxels (mask and a legal label) of logits [B,X,Y,Z,C]."""
    classes = logits.shape[-1]
    validate_free_weight(free_target_weight, free_id, classes)
    if len(positive_weights) != classes or len(negative_weights) != classes:
        raise ValueError('Class weight vocabulary mismatch')
    valid = mask.bool() & (labels >= 0) & (labels < classes)
    if not valid.any():
        return logits.float().sum() * 0
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logp, logq = stable_log_probabilities(logits[valid].float())
        y = labels[valid].long()
        logpt = logp.gather(1, y[:, None]).squeeze(1)
        positive = logp.new_tensor(positive_weights)[y] * (-torch.expm1(logpt)).square() * (-logpt)
        negative = logp.new_tensor(negative_weights) * logp.exp().square() * (-logq)
        negative = negative.scatter(1, y[:, None], 0.).sum(-1)
        value = positive + negative
        if free_target_weight != 1.:
            value = value * torch.where(y == free_id, free_target_weight, 1.)
        return value.mean() * scale


def grid_loss(model, logits, labels, mask):
    """L_grid of one target frame; the Triton kernel on CUDA when ``model.fused_loss``."""
    spec = model.occupancy_objective
    if getattr(model, 'fused_loss', False) and logits.is_cuda:
        from .kernels.focal_loss import fused_focal_field_loss as loss
    else:
        loss = focal_field_loss
    return loss(logits, labels, mask, spec['positive_weights'], spec['negative_weights'],
                spec['train_loss_scale'], spec['free_target_weight'], spec['free_id'])


def frame_loss(model, batch, sup, future, current, ray_logits=None):
    """Loss of one supervised frame (call under the forward's autocast).

    ``future``: forecast items at ``batch['horizons']``; ``current``: current readout; ``ray_logits``:
    callable returning the writer's ray-head logits [N,P,D] of the current capture group (camera writer).
    Returns (loss, ray term or None); the ray term is detached, for logging.
    """
    horizons = list(batch.get('horizons', [1, 2, 3]))
    if len(future) != len(horizons) or (future and sup['labels'].shape[1] != len(horizons)):
        raise ValueError('Prediction/target horizon mismatch')
    terms = {}
    for i, (h, prediction) in enumerate(zip(horizons, future)):
        if h in terms:
            raise ValueError('Repeated horizon')
        terms[h] = grid_loss(model, prediction['occupancy'], sup['labels'][:, i], sup['mask_camera'][:, i])
    current_term = grid_loss(model, current['occupancy'], sup['current_labels'], sup['current_mask_camera'])
    ray = None
    writer = model.writer
    if getattr(writer, 'depth_bins', 0):
        b = model.config.bounds
        hw = tuple(batch['rgb'].shape[-2:])
        device = current['occupancy'].device
        occupancy_target, valid = ray_depth.render_ray_targets(
            sup['current_labels'][0], batch['current_world_from_ego'][0].to(device),
            batch['image_from_world'][0, -1].to(device), hw, (hw[0]//16, hw[1]//16), writer.depth_bins,
            writer.depth_bin_size, free_id=model.config.free_id, bounds=tuple(b),
            voxel=(b[3]-b[0])/model.config.occupancy_shape[0])
        pred = ray_logits()
        objective = model.occupancy_objective
        ray_sum, ray_n = ray_depth.ray_focal_loss(pred, occupancy_target.reshape(pred.shape),
                                                  valid.reshape(pred.shape), objective['field_priors'],
                                                  model.config.free_id, objective['free_target_weight'])
        # The ray view enters as its own mean over the valid sections, next to the grid term.
        current_term = current_term+ray_sum/max(ray_n, 1)
        ray = (ray_sum/max(ray_n, 1)).detach()
    weight = model.occupancy_objective['current_weight']
    # Mean over the supervised horizons plus the weighted current-frame terms. For 1/2/3 s the mean is
    # written as sum/3, the exact arithmetic the released models were trained with.
    if horizons == [1, 2, 3]:
        loss = torch.stack([terms[h] for h in horizons]).sum()/3+weight*current_term
    else:
        loss = torch.stack([terms[h] for h in horizons]).mean()+weight*current_term
    return loss, ray
