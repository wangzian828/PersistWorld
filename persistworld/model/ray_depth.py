"""Ray-depth sections of the camera writer, and their supervision.

All sub-cells along a pixel ray project to the same image location and sample the same feature.
Each patch ray is split into depth sections of ``bin_size`` (0.8 m). A light head predicts, per
patch, the occupancy o_k of every section; the first-surface probability is rendered as
p_k = o_k * prod_{j<k} (1 - o_j), so the probability that the ray hits no surface is the rest.
Every writer sample receives the log-probability of its section as an attention bias, and a
geometry embedding (its log camera depth, image position and that log-probability) added to its
key and value. All additions are exact no-ops at initialization.

The ray view is supervised with the same occupancy labels, rendered along the patch rays.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _zero_mlp(inputs, dim):
    mlp = nn.Sequential(nn.Linear(inputs, dim), nn.GELU(), nn.Linear(dim, dim))
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


def install_ray_depth(writer, feature_dim, bins, bin_size=.8):
    """Add the ray head, its bias scale and the geometry embeddings to a camera writer."""
    writer.depth_bins = bins
    writer.depth_bin_size = bin_size
    writer.depth_records = []
    writer.record_depth = False
    device = next(writer.parameters()).device
    hidden = 256
    writer.depth_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden),
                                      nn.GELU(), nn.Linear(hidden, bins)).to(device)
    nn.init.zeros_(writer.depth_head[-1].weight)
    nn.init.zeros_(writer.depth_head[-1].bias)
    # Hazard 1/(D+1-k) renders a uniform first-surface distribution: an exact no-op at init.
    k = torch.arange(bins, dtype=torch.float32)
    hazard = 1/(bins+1-k)
    with torch.no_grad():
        writer.depth_head[-1].bias.copy_(torch.log(hazard/(1-hazard)).to(device))
    writer.depth_scale = nn.Parameter(torch.ones((), device=device))
    writer.geometry_key = _zero_mlp(4, writer.dim).to(device)
    writer.geometry_value = _zero_mlp(4, writer.dim).to(device)
    return writer


def depth_volume(writer, keys, patch_grid):
    """Image features [B,T,N,P,C] -> per-image log-probability volume [B*T*N,1,D,h,w]."""
    if getattr(writer, 'ablate_depth', False):
        # abl_no_depth: the ray logits are still supervised (ray loss), but nothing enters the writing path.
        if writer.record_depth:
            writer.depth_records.append(writer.depth_head(keys.float()))
        return None
    b, t, n, p, _ = keys.shape
    h, w = patch_grid
    logits = writer.depth_head(keys.float())
    if writer.record_depth:
        writer.depth_records.append(logits)
    bins = writer.depth_bins
    logp = render_visibility(logits) + math.log(bins+1)
    return logp.reshape(b*t*n, h, w, bins).permute(0, 3, 1, 2)[:, None].contiguous()


def sample_geometry(writer, grid, depth, volume, b, t, cameras, count):
    """grid [B,T,N,L,P,2], depth [B,T,N,L,P] -> attention bias [B,L,T*N*P] and geometry features [B,L,T*N*P,4],
    in the writer's token order (frame, camera, sample)."""
    pairs = grid.shape[-2]
    with torch.autocast(device_type=grid.device.type, enabled=False):
        grid = grid.float()
        depth = depth.float()
        span = writer.depth_bins*writer.depth_bin_size
        z = 2*depth.clamp_min(0)/span-1
        coords = torch.cat([grid, z[..., None]], -1).reshape(b*t*cameras, count*pairs, 1, 1, 3)
        sampled = F.grid_sample(volume.float(), coords, mode='bilinear', padding_mode='border', align_corners=False)
        logp = sampled.reshape(b, t, cameras, count, pairs).clamp(-20., 20.)
        log_depth = 2*torch.log(depth.clamp(.5, 80.)/.5)/math.log(160.)-1
        parts = [log_depth[..., None], grid.clamp(-1.5, 1.5), logp[..., None]/5]
        features = torch.cat(parts, -1).permute(0, 3, 1, 2, 4, 5).reshape(b, count, t*cameras*pairs, -1)
        tokens = logp.permute(0, 3, 1, 2, 4).reshape(b, count, t*cameras*pairs)
        bias = writer.depth_scale.float()*tokens
    return bias, features


def render_visibility(logits):
    """Ray-section occupancy logits [...,D] -> log first-surface probability [...,D]."""
    x = logits.float()
    free = F.logsigmoid(-x)
    before = torch.cumsum(free, -1)-free                     # exclusive prefix: sum_{j<k} log(1-o_j)
    return F.logsigmoid(x)+before


def render_ray_targets(labels, world_from_ego, image_from_world, image_hw, patch_grid, bins, bin_size, *,
                       free_id, bounds, voxel=.4, step=.2, subrays=2):
    """Section occupancy targets and validity [N,h,w,D] of one camera group from a label grid.

    labels [X,Y,Z] in the label ego frame; image_from_world [N,4,4] projects world points to
    pixels with the camera z-depth as third coordinate. Each patch is sampled by subrays x subrays
    rays marched in steps of ``step``; a ray stops at its first occupied voxel and is no longer
    counted after it leaves the label grid.
    """
    device = image_from_world.device
    h, w = patch_grid
    H, W = image_hw
    n = image_from_world.shape[0]
    max_depth = bins*bin_size
    offsets = (torch.arange(subrays, device=device, dtype=torch.float64)+.5)/subrays
    py = (torch.arange(h, device=device, dtype=torch.float64)[:, None]+offsets)*(H/h)
    px = (torch.arange(w, device=device, dtype=torch.float64)[:, None]+offsets)*(W/w)
    v = py.reshape(h, 1, subrays, 1).expand(h, w, subrays, subrays).reshape(-1)-.5
    u = px.reshape(1, w, 1, subrays).expand(h, w, subrays, subrays).reshape(-1)-.5
    d = torch.arange(step, max_depth+1e-6, step, device=device, dtype=torch.float64)
    rays = u.numel()
    pix = torch.stack([u[:, None]*d, v[:, None]*d, d.expand(rays, -1), torch.ones(rays, d.numel(),
                       device=device, dtype=torch.float64)], -1)          # R,S,4
    ego_from_world = torch.linalg.inv(world_from_ego.double())
    lo = torch.tensor(bounds[:3], device=device, dtype=torch.float64)
    shape = torch.tensor(labels.shape, device=device)
    hit_depth = torch.zeros(n, rays, device=device)
    exit_depth = torch.zeros(n, rays, device=device)
    for cam in range(n):
        world_from_image = torch.linalg.inv(image_from_world[cam].double())
        points = pix@(ego_from_world@world_from_image).T                  # R,S,4
        idx = torch.floor((points[..., :3]-lo)/voxel).long()
        inside = ((idx >= 0) & (idx < shape)).all(-1)
        idx = idx.clamp_min(0).minimum(shape-1)
        occupied = (labels[idx[..., 0], idx[..., 1], idx[..., 2]] != free_id) & inside
        # A ray that left the convex grid never re-enters: count only hits before its first exit.
        entered = torch.cumsum(inside.int(), -1) > 0
        left = entered & ~inside
        stop = torch.cumsum(left.int(), -1) > 0
        occupied &= ~stop
        hit = occupied.any(-1)
        first = occupied.int().argmax(-1)
        depth = d[first]
        seen = inside & ~stop
        exit_depth[cam] = torch.where(seen.any(-1),
                                      d[(seen.int()*torch.arange(1, d.numel()+1, device=device)).argmax(-1)],
                                      torch.zeros_like(d[:1])).float()
        hit_depth[cam] = torch.where(hit, depth, torch.zeros_like(depth)).float()
    return ray_occupancy_targets(hit_depth.reshape(n, h, w, subrays*subrays), bins, bin_size,
                                 exit_depth.reshape(n, h, w, subrays*subrays))


def ray_occupancy_targets(hit_depth, bins, bin_size, exit_depth):
    """Per-subray first hit (0 = none) and grid-exit depth [N,h,w,S] -> soft section occupancy targets and
    validity [N,h,w,D]: free before the first surface, occupied at it, unseen (invalid) behind it; rays
    without a surface are free until they leave the label grid."""
    k = torch.arange(bins, device=hit_depth.device)
    hit_bin = torch.where(hit_depth > 0, (hit_depth/bin_size).long().clamp_max(bins-1),
                          torch.full_like(hit_depth, -1, dtype=torch.long))
    exit_bin = (exit_depth/bin_size).long().clamp(0, bins)
    last = torch.where(hit_bin >= 0, hit_bin, exit_bin-1)[..., None]           # last seen section
    valid = k <= last                                                          # N,h,w,S,D
    occupied = (k == hit_bin[..., None])
    count = valid.sum(-2)
    target = (occupied & valid).sum(-2).float()/count.clamp_min(1)
    return target, count > 0


def ray_focal_loss(logits, target, valid, field_priors, free_id, free_target_weight=1.):
    """The focal occupancy loss on the two-class (free/occupied) ray field, weights from the same
    class priors. Returns (sum, count) over the valid sections."""
    p_free = float(field_priors[free_id])
    p_occ = 1-p_free
    pos = [p_free**-.5, p_occ**-.5]
    neg = [(1-p_free)**-.5, (1-p_occ)**-.5]
    norm = p_free*pos[0]+(1-p_free)*neg[0]+p_occ*pos[1]+(1-p_occ)*neg[1]
    w_occ = (pos[1]+neg[0])/norm
    w_free = (neg[1]+pos[0])/norm
    with torch.autocast(device_type=logits.device.type, enabled=False):
        x = logits.float()[valid]
        y = target.float()[valid]
        p = torch.sigmoid(x)
        value = (w_occ*y*(1-p).square()*(-F.logsigmoid(x))
                 + w_free*(1-y)*p.square()*(-F.logsigmoid(-x)))
        value = value*(free_target_weight*(1-y)+y)
    return value.sum(), value.numel()
