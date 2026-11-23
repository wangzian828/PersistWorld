"""Camera writer: writes the evidence of one capture group into the state carried to its capture time.

For a sub-cell with center x, the writer samples keys and values bilinearly at the projection of x
into each camera, shifted by a small image-plane offset predicted from the sub-cell's feature, and
adds a learned camera embedding. Ray-depth sections (ray_depth.py) add a geometry embedding to the
key and value and the log-probability of the sample's section as an attention bias. The sub-cell
feature attends, with a single query, to its samples from all cameras and to a null token with zero
key and value. A gate computed from the prior and the candidate controls the residual update and is
zero for sub-cells that no camera sees, so those keep the state carried by the evolver.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import host_sync as hs
from . import ray_depth
from .geometry import project_homogeneous
from .state import SceneField, pack_subcells, subcell_points, unpack_subcells


class CameraWriter(nn.Module):
    """Writer on the 0.8 m sub-cells of the packed state (feature width ``packed_dim // 8``)."""
    def __init__(self, feature_dim, packed_dim, cell_size, layers=2, depth_bins=75, depth_bin_size=.8):
        super().__init__()
        if packed_dim < 16 or packed_dim % 8:
            raise ValueError('Packed dimension must be a multiple of eight >= 16')
        dim = packed_dim//8
        self.dim = dim
        self.heads = 1
        self.key = nn.Linear(feature_dim, dim)
        self.value = nn.Linear(feature_dim, dim)
        self.tag = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.query = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.output = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.GELU(),
                                               nn.Linear(2*dim, dim))
                                 for _ in range(layers)])
        self.offset = nn.Sequential(nn.Linear(dim, dim//2), nn.GELU(), nn.Linear(dim//2, 2))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.gate = nn.Sequential(nn.Linear(2*dim, dim), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -1)
        self.packed_dim = packed_dim
        self.cell_size = tuple(cell_size)
        ray_depth.install_ray_depth(self, feature_dim, depth_bins, depth_bin_size)
        # Execution switches (execution.configure_execution) and ablation switches (build.build_model).
        self.query_chunk = 32768
        self.checkpoint_chunks = False
        self.fused_evidence = False
        self.activation_target_bytes = 0
        self.ablate_attention = False   # ablation abl_mean_writing
        self.ablate_depth = False       # ablation abl_no_depth: ray depth trained, but not used for writing

    def _fused(self, prior):
        """Whether the Triton evidence kernels run (CUDA, one scene)."""
        return self.fused_evidence and prior.is_cuda and prior.shape[0] == 1

    def _chunk(self, prior, xyz, key_maps, image_from_scene, image_hw, fresh, tags, patch_grid, depth_volume=None):
        """Write one chunk of sub-cells: sample their evidence, attend over it and gate the update.

        prior [B,L,d] sub-cell features, xyz their centres [L,3] (or [B,L,3]) in the scene frame, ``fresh`` [B,T]
        the groups to write; returns the posterior [B,L,d] and whether any camera sees each sub-cell [B,L].
        """
        b, t, cameras = image_from_scene.shape[:3]
        count, d = prior.shape[1:]
        h, w = patch_grid
        with torch.autocast(device_type=prior.device.type, enabled=False):
            homogeneous = torch.cat([xyz.float(), torch.ones_like(xyz[..., :1]).float()], -1)
            base = (project_homogeneous(image_from_scene, homogeneous) if xyz.ndim == 2 else
                    torch.einsum('btnij,blj->btnli', image_from_scene.float(), homogeneous))
            projected = base[..., None, :]
            depth = projected[..., 2]
            uv = projected[..., :2]/depth.clamp_min(1e-5)[..., None]
            grid = (uv+.5)/hs.const([image_hw[1], image_hw[0]], uv)*2-1
        offset = self.offset(prior[:, None].expand(-1, t, -1, -1)).tanh()
        grid = grid+offset[:, :, None, :, None]*hs.const([2/w, 2/h], grid)
        valid = (depth > 1e-5) & (grid.abs() < 1).all(-1) & fresh[:, :, None, None, None]
        valid = valid.permute(0, 3, 1, 2, 4)  # B,L,T,camera,sample
        fused = self._fused(prior)
        if fused:
            # key_maps is channels-last [B*T*N,h,w,2d]; the output is already token-major.
            from ..kernels.fused_evidence import sample_tokens
            keys, values = sample_tokens(key_maps, grid.reshape(b*t*cameras, count, 1, 2).float(),
                                         tags.reshape(t*cameras, d), prior.dtype)
            keys, values = keys[None], values[None]
        else:
            keys, values = self._sample_evidence(key_maps, grid.reshape(b*t*cameras, count, 1, 2).float(),
                                                 b, t, cameras, count, d, tags, prior)
        bias = torch.zeros(valid.shape, device=valid.device, dtype=torch.float32
                           ).masked_fill(~valid, -torch.inf).reshape(b, count, -1)
        mean_only = self.ablate_attention
        if not self.ablate_depth:
            depth_bias, geometry = ray_depth.sample_geometry(self, grid, depth, depth_volume, b, t, cameras, count)
            if not mean_only:
                bias = bias+depth_bias.to(bias.dtype)
            geometry = geometry.to(keys.dtype)
            keys = keys+self.geometry_key(geometry).to(keys.dtype)
            values = values+self.geometry_value(geometry).to(values.dtype)
        null = torch.zeros_like(bias[:, :, :1])
        if mean_only:
            # Ablation: plain average of the visible samples (zero query, no depth bias); the null
            # token only covers unseen cells. -1e4 (not -inf) keeps the fused kernel's running max finite.
            null = torch.where(valid.reshape(b, count, -1).any(-1, keepdim=True),
                               torch.full_like(null, -1e4), torch.zeros_like(null))
        candidate = prior
        if fused:
            from ..kernels.fused_evidence import attend
            for norm, qproj, output, ff in zip(self.norm, self.query, self.output, self.ff):
                query = qproj(norm(candidate)).reshape(count, d)
                if mean_only:
                    query = query*0
                evidence = attend(query, keys[0], values[0], bias[0], null[0, :, 0]).reshape(b, count, d)
                candidate = candidate+output(evidence)
                candidate = candidate+ff(candidate)
            seen = valid.flatten(2).any(-1)
            gate = self.gate(torch.cat([prior, candidate], -1))*seen[..., None]
            return prior+gate*(candidate-prior), seen
        keys = torch.cat([keys, torch.zeros_like(keys[:, :, :1])], 2)
        values = torch.cat([values, torch.zeros_like(values[:, :, :1])], 2)
        bias = torch.cat([bias, null], -1)
        tokens = t*cameras+1
        kh = keys.reshape(b*count, tokens, self.heads, d//self.heads).transpose(1, 2)
        vh = values.reshape(b*count, tokens, self.heads, d//self.heads).transpose(1, 2)
        for norm, qproj, output, ff in zip(self.norm, self.query, self.output, self.ff):
            query = qproj(norm(candidate)).reshape(b*count, self.heads, 1, d//self.heads)
            if mean_only:
                query = query*0
            evidence = F.scaled_dot_product_attention(query, kh, vh, attn_mask=bias.reshape(b*count, 1, 1, tokens),
                                                      dropout_p=0.).reshape(b, count, d)
            candidate = candidate+output(evidence)
            candidate = candidate+ff(candidate)
        seen = valid.flatten(2).any(-1)
        gate = self.gate(torch.cat([prior, candidate], -1))*seen[..., None]
        return prior+gate*(candidate-prior), seen

    def _sample_evidence(self, key_maps, sampling_grid, b, t, cameras, count, d, tags, prior):
        """PyTorch reference of kernels.fused_evidence.sample_tokens: keys (with camera tags) and values [B,L,T*N,d]."""
        # One grid traversal for the packed K and V maps.
        sampled = F.grid_sample(key_maps, sampling_grid, align_corners=False).to(prior.dtype)
        kv = sampled.squeeze(-1).reshape(b, t, cameras, 2, d, count, 1).unbind(3)
        keys, values = [x.permute(0, 4, 1, 2, 5, 3).reshape(b, count, t*cameras, d) for x in kv]
        keys = (keys.reshape(b, count, t, cameras, 1, d)+tags).reshape(b, count, t*cameras, d)
        return keys, values

    def _project_evidence(self, features, patch_grid):
        """Key and value projections of the patch features, packed into one map per image."""
        b, t, cameras = features.shape[:3]
        h, w = patch_grid

        def maps(linear):
            return (linear(features).reshape(b, t, cameras, h, w, self.dim)
                    .permute(0, 1, 2, 5, 3, 4).reshape(b*t*cameras, self.dim, h, w).float())

        if self._fused(features):
            # Channels-last packed maps for the fused sampler; same values as the reference.
            packed = torch.cat([self.key(features), self.value(features)], -1)
            return packed.reshape(b*t*cameras, h, w, 2*self.dim)
        return torch.cat((maps(self.key), maps(self.value)), 1)

    def _checkpoint_chunk(self, prior):
        """Recompute a chunk in backward when configured, or when allocated memory exceeds the target."""
        recompute = self.checkpoint_chunks
        if self.training and self.activation_target_bytes and prior.device.type == 'cuda':
            recompute = recompute or torch.cuda.memory_allocated(prior.device) >= self.activation_target_bytes
        return recompute

    def forward(self, state, features, image_from_scene, times, image_hw, patch_grid, reference_points):
        """Write the patch features [B,T,N,P,C] of the capture groups at ``times`` [B,T] into the packed ``state``.

        ``image_from_scene`` [B,T,N,4,4] maps the scene frame to pixels and ``reference_points`` are the cell
        centres in that frame. Groups older than the state's last observation are not written again. Returns
        the posterior and the per-cell coverage (cells that some camera sees)."""
        if state.features.shape[-1] != self.packed_dim:
            raise ValueError('Writer requires the packed state')
        fine = unpack_subcells(state.features)
        points = subcell_points(reference_points, self.cell_size)
        b, t, cameras, patches, _ = features.shape
        if patches != math.prod(patch_grid):
            raise ValueError('Evidence shape mismatch')
        if t < 1 or times.shape != (b, t) or (hs.checks() and torch.any(times[:, -1] != state.time)):
            raise ValueError('State must be at the observed prefix end')
        if (hs.checks() and torch.any(times[:, 1:] < times[:, :-1])) or self.query_chunk <= 0:
            raise ValueError('Expected ordered times and positive query chunk')
        fresh = times >= state.last_observation_time[:, None]
        key_maps = self._project_evidence(features, patch_grid)
        volume = ray_depth.depth_volume(self, features, patch_grid)
        prior = fine.flatten(1, 3)
        batched_points = points.ndim == 5
        points = points.reshape(b, -1, 3) if batched_points else points.reshape(-1, 3)
        if points.shape[-2] != prior.shape[1]:
            raise ValueError('Reference point count mismatch')
        camera = torch.arange(cameras, device=prior.device, dtype=prior.dtype)/max(cameras-1, 1)
        tags = self.tag(camera[None, None, :, None].expand(b, t, -1, 1)).reshape(b, 1, t, cameras, 1, self.dim)
        pieces, coverage = [], []
        kind = prior.device.type
        for start in range(0, prior.shape[1], self.query_chunk):
            args = (prior[:, start:start+self.query_chunk],
                    (points[:, start:start+self.query_chunk] if batched_points
                     else points[start:start+self.query_chunk]),
                    key_maps, image_from_scene, image_hw, fresh, tags, patch_grid)
            # Uncached autocast casts: every chunk returns its parameter gradient to the FP32 leaf.
            context = torch.autocast(kind, enabled=torch.is_autocast_enabled(kind),
                                     dtype=torch.get_autocast_dtype(kind), cache_enabled=False)
            recompute = self._checkpoint_chunk(prior)
            with context:
                value, seen = (checkpoint(self._chunk, *args, use_reentrant=False, depth_volume=volume)
                               if self.training and recompute else self._chunk(*args, depth_volume=volume))
            pieces.append(value)
            coverage.append(seen)
        features = torch.cat(pieces, 1).reshape_as(fine)
        seen = torch.cat(coverage, 1).reshape(features.shape[:-1])
        posterior = SceneField(pack_subcells(features), state.world_from_ego, state.time,
                               torch.maximum(state.last_observation_time, times[:, -1]), state.updates+1)
        # Coverage records a write opportunity of the 1.6 m cell, not visibility or confidence.
        coverage = pack_subcells(seen[..., None]).any(-1)
        return posterior, coverage
