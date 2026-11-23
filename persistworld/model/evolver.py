"""Evolver: advances the world state from the time of its estimate to a target time.

Inputs are the state window around the ego at the target time and the ego poses since the last
reset up to the target pose. Motion tokens are formed from the relative SE(3) transforms between
consecutive poses and the time steps; every cell attends only to the motion that follows its
own estimate and is conditioned on the age of its estimate and of its last observation. A
context of the observed state (axial blocks over all stored cells) is computed once per state
update, shared by every advance from it, and added before the residual blocks. Residual blocks
with adaptive layer normalization apply cross-attention to the motion tokens and
self-attention along one lattice axis. The output is a change added to the state, scaled by
the elapsed time, so the evolver is the identity when no time passes.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import attention
from . import geometry
from . import host_sync as hs
from ..kernels.fused_norm import normalized_modulation


class EvolverBlock(nn.Module):
    """AdaLN residual block: motion cross-attention, then self-attention along lattice axis ``axis``."""
    def __init__(self, dim, heads, axis):
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.motion = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.spatial = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim, elementwise_affine=False), nn.Linear(dim, 4*dim), nn.GELU(),
                                nn.Linear(4*dim, dim))
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6*dim))
        nn.init.normal_(self.cond[-1].weight, std=1e-3)
        nn.init.zeros_(self.cond[-1].bias)

    def forward(self, x, motion, cell_condition=None, motion_mask=None):
        shape = x.shape
        b = shape[0]
        d = shape[-1]
        normalized = self.norm(x)
        c = attention.bounded_math_multihead(self.motion, normalized.reshape(b, -1, d), motion, chunk=None,
                                             attn_mask=motion_mask).reshape(shape)
        if cell_condition is not None:
            c = c+cell_condition
        shift, scale, gate, ff_shift, ff_scale, ff_gate = self.cond(c).chunk(6, -1)
        h = normalized*(1+scale)+shift
        # Move the selected spatial axis next to D; the remaining axes become batches.
        axis = self.axis+1
        order = [0]+[i for i in (1, 2, 3) if i != axis]+[axis, 4]
        h = h.permute(order)
        permuted = h.shape
        h = h.reshape(-1, shape[axis], d)
        h = attention.bounded_math_multihead(self.spatial, h, h, chunk=None, attn_mask=None).reshape(permuted)
        inverse = [order.index(i) for i in range(5)]
        x = x+gate*h.permute(inverse)
        # ff[0] (a LayerNorm without affine parameters) runs inside the fused normalize-and-modulate kernel.
        ff = normalized_modulation(x, ff_scale, ff_shift, self.norm.eps, self.ff[0].eps)
        for layer in self.ff[1:]:
            ff = layer(ff)
        return x+ff_gate*ff


class ContextBlock(nn.Module):
    """Self-attention along one lattice axis over the stored cells; empty cells are neither keys nor updated."""
    def __init__(self, dim, heads, axis):
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(dim)
        self.spatial = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))

    def forward(self, x, valid):
        axis = self.axis + 1
        order = [0] + [i for i in (1, 2, 3) if i != axis] + [axis, 4]
        h = self.norm(x).permute(order)
        shape = h.shape
        h = h.reshape(-1, shape[-2], shape[-1])
        keys = valid.permute(order[:-1]).reshape(-1, shape[-2])
        mask = (~keys[:, None, :]).expand(-1, shape[-2], -1)
        h = attention.bounded_math_multihead(self.spatial, h, h, chunk=None, attn_mask=mask).reshape(shape)
        x = x + h.permute([order.index(i) for i in range(5)])
        return torch.where(valid[..., None], x + self.ff(x), 0.)


class Evolver(nn.Module):
    """Motion-conditioned residual blocks that advance a state window in time, and the observed-context blocks."""
    def __init__(self, dim=384, heads=6, layers=3, context_layers=3):
        super().__init__()
        self.checkpoint_blocks = False   # execution option (execution.configure_execution)
        self.motion_embed = nn.Sequential(nn.Linear(15, dim), nn.GELU(), nn.Linear(dim, dim))
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([EvolverBlock(dim, heads, i % 3) for i in range(layers)])
        self.out = nn.Linear(dim, dim)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)
        # Normalize only the learned increment input; the identity carry stays intact.
        self.increment_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.age_embed = nn.Sequential(nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.context_blocks = nn.ModuleList([ContextBlock(dim, heads, i) for i in range(context_layers)])

    def _checkpoint_layer(self):
        """Whether the blocks recompute their activations in backward."""
        return self.training and self.checkpoint_blocks

    def forward(self, window, world_from_ego, times, target_time, position_features, *, context_delta=None):
        """Advance ``window`` (WorldWindow) to ``target_time`` along the pose history ``world_from_ego``/``times``.

        ``position_features``: ego-relative cell positions in [-1, 1]; ``context_delta``: the shared observed
        context cropped to the window.
        """
        batch = window.features.shape[0]
        if batch != 1:
            raise ValueError('The evolver advances one scene window at a time')
        if times.ndim != 2 or times.shape[0] != batch or times.shape[1] == 0:
            raise ValueError('Nonempty ordered motion history required')
        if hs.checks() and (torch.any(times[:, 1:] < times[:, :-1])
                            or not torch.allclose(times[:, -1], target_time)):
            raise ValueError('Motion history must end at the query time')
        elapsed = target_time[:, None, None, None]-window.state_time
        if hs.checks() and torch.any(elapsed < -1e-5):
            raise ValueError('Cannot read a future estimate into the past')
        elapsed = elapsed.clamp_min(0)
        prev = torch.cat([world_from_ego[:, :1], world_from_ego[:, :-1]], 1).float()
        relative = geometry.solve(prev, world_from_ego)
        dt = torch.diff(torch.cat([times[:, :1], times], 1), dim=1)
        age = times-times[:, :1]
        phase = (torch.arange(times.shape[1], device=times.device, dtype=times.dtype)[None, :, None]
                 .expand(batch, -1, -1)/16)
        motion = torch.cat([relative[:, :, :3, :3].flatten(-2), relative[:, :, :3, 3]/10,
                            dt[..., None], age[..., None], phase], -1)
        motion = self.motion_embed(motion.to(window.features.dtype))
        observed = torch.isfinite(window.observed_time)
        evidence_age = torch.where(observed,
                                   (target_time[:, None, None, None]-window.observed_time).clamp_min(0), 0)
        metadata = torch.stack([torch.log1p(elapsed), torch.log1p(evidence_age), observed.float(),
                                window.valid.float()], -1)
        condition = self.age_embed(metadata.to(window.features.dtype))
        # A cell attends to the motion after its own estimate; every row keeps the final token.
        mask = times[0][None, :] < window.state_time.flatten()[:, None]-1e-5
        position = self.position(position_features.to(window.features.dtype))
        h = window.features+position[None]
        if context_delta is not None:
            if context_delta.shape != h.shape:
                raise ValueError('Context/query shape mismatch')
            h = h+context_delta
        for block in self.blocks:
            if self._checkpoint_layer():
                h = checkpoint(block, h, motion, condition, mask, use_reentrant=False)
            else:
                h = block(h, motion, condition, mask)
        result = window.features+self.out(self.increment_norm(h))*elapsed[..., None]
        result = torch.where((elapsed == 0)[..., None], window.features, result)
        return window.replace(result, state_time=target_time[:, None, None, None].expand_as(window.state_time))

    def encode_context(self, window, reference_time, position):
        """Context residual of the observed state window (zero at cells without a stored estimate)."""
        elapsed = reference_time - window.state_time
        if hs.checks() and torch.any(window.valid & (elapsed < -1e-5)):
            raise ValueError('Context contains a future state estimate')
        observed = torch.isfinite(window.observed_time) & window.valid
        if hs.checks() and torch.any(observed & (window.observed_time > reference_time + 1e-5)):
            raise ValueError('Context contains future evidence')
        evidence_age = torch.where(observed, reference_time - window.observed_time, 0.)
        metadata = torch.stack([torch.log1p(elapsed.clamp_min(0)), torch.log1p(evidence_age.clamp_min(0)),
                                observed.float(), window.valid.float()], -1)
        base = window.features + self.position(position.to(window.features.dtype))[None]
        base = base + self.age_embed(metadata.to(window.features.dtype))
        base = torch.where(window.valid[..., None], base, 0.)
        h = base
        for block in self.context_blocks:
            if self._checkpoint_layer():
                h = checkpoint(block, h, window.valid, use_reentrant=False)
            else:
                h = block(h, window.valid)
        return torch.where(window.valid[..., None], h - base, 0.)


def parameter_stamp(model):
    """Identity and in-place version of every trainable parameter (detects optimizer steps)."""
    return tuple((id(p), p._version) for p in model.parameters() if p.requires_grad)


@dataclass(frozen=True)
class ObservedContext:
    """Context residual of one session state, valid until the state or the parameters change."""
    delta: object
    origin: tuple
    cell_size: tuple
    memory_id: int
    memory_revision: int
    reference_time: float
    session_stamp: tuple
    parameters: tuple
    training: bool
    grad_enabled: bool

    def crop(self, model, session, window):
        """The context residual on the cells of ``window`` (zero outside the observed extent)."""
        if (id(session.memory) != self.memory_id or session.memory.revision != self.memory_revision
                or model.observation_stamp(session) != self.session_stamp
                or parameter_stamp(model) != self.parameters
                or model.training != self.training or torch.is_grad_enabled() != self.grad_enabled):
            raise ValueError('Stale or foreign observed context')
        if tuple(window.cell_size) != self.cell_size:
            raise ValueError('Context and query lattice differ')
        out = torch.zeros_like(window.features)
        if self.delta is None:
            return out
        lo = tuple(max(a, b) for a, b in zip(self.origin, window.origin))
        hi = tuple(min(a+n, b+m)
                   for a, n, b, m in zip(self.origin, self.delta.shape[1:4], window.origin, window.shape))
        if any(b <= a for a, b in zip(lo, hi)):
            return out
        source = tuple(slice(a-o, b-o) for a, b, o in zip(lo, hi, self.origin))
        target = tuple(slice(a-o, b-o) for a, b, o in zip(lo, hi, window.origin))
        out[(slice(None), *target, slice(None))] = self.delta[(slice(None), *source, slice(None))]
        return out


def prepare_context(model, session):
    """Observed context of the session state; its extent and encoding depend on observations only."""
    if session.times.shape[1] > 64:
        raise ValueError('The observed context expects at most 64 observed frames')
    memory = session.memory
    origin = (0, 0, 0)
    delta = None
    if memory.tiles:
        origin = tuple(min(k[i] for k in memory.tiles)*memory.tile_shape[i] for i in range(3))
        upper = tuple((max(k[i] for k in memory.tiles)+1)*memory.tile_shape[i] for i in range(3))
        shape = tuple(b-a for a, b in zip(origin, upper))
        # Keep the stored state precision; projections still use autocast.
        initial = next(model.initial.parameters()).new_zeros((1, *shape, model.config.dim))
        window = memory.read(origin, initial, session.last_time)
        scene_from_ego = geometry.solve(session.world_from_scene, session.poses[0, -1])
        points = geometry.transform_points(geometry.inverse(scene_from_ego),
                                           window.points().reshape(-1, 3)).reshape(*shape, 3)
        low = hs.const(list(model.config.bounds[:3]), points)
        span = hs.const(list(model.config.bounds[3:]), points) - low
        delta = model.evolver.encode_context(window, session.last_time, 2*(points-low)/span-1)
    return ObservedContext(delta, origin, tuple(memory.cell_size), id(memory), memory.revision,
                           session.last_time, model.observation_stamp(session), parameter_stamp(model),
                           model.training, torch.is_grad_enabled())
