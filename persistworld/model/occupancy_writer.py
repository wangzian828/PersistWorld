"""Occupancy writer: the same world state written from an occupancy grid instead of images.

Used by PersistWorld-GT (ground-truth occupancy input) and PersistWorld-STC (occupancy predicted
by STCOcc). Only the writer changes; the evolver, the reader and the losses are those of the
camera model.

Every packed cell has eight 0.8 m sub-cells, each covering 2x2x2 Occ3D voxels of 0.4 m. For each
input frame the eight voxel centres of every sub-cell are moved into that frame's ego grid with
the ego poses, the one-hot labels are sampled trilinearly (exact at alignment) and the 8x18 child
histogram is embedded. Per sub-cell, attention over the input frames (with a null token) and a
gated residual write the evidence into the prior, as the camera writer does for images.

Geometry: a query passes ``image_from_world = ego_from_world`` (one "camera" per frame), so the
model's usual composition gives ego_t_from_scene.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import host_sync as hs
from .geometry import transform_points
from .state import SceneField, pack_subcells, subcell_points, unpack_subcells


class OccupancyInput(nn.Module):
    """Stand-in for the image encoder: label grids pass through unchanged."""
    def __init__(self, num_classes):
        super().__init__()
        self.feature_dim = num_classes

    def encode(self, grids):
        return {'features': grids, 'patch_grid': (1, 1)}

    def writer_input(self, encoded):
        return encoded

    def forward(self, grids):
        return self.encode(grids)


class OccupancyWriter(nn.Module):
    """Writes occupancy grids [B,T,1,X,Y,Z] (class ids) into the packed state."""

    def __init__(self, packed_dim, cell_size, num_classes=18, bounds=(-40., -40., -1., 40., 40., 5.4),
                 grid_shape=(200, 200, 16), layers=2):
        super().__init__()
        if packed_dim % 8:
            raise ValueError('Packed dimension must be a multiple of eight')
        d = self.dim = packed_dim//8
        self.packed_dim = packed_dim
        self.cell_size = tuple(cell_size)
        self.classes = num_classes
        self.bounds = tuple(bounds)
        self.grid_shape = tuple(grid_shape)
        self.query_chunk = 32768          # execution options (execution.configure_execution)
        self.checkpoint_chunks = False
        self.embed = nn.Sequential(nn.Linear(8*num_classes, 2*d), nn.GELU(), nn.Linear(2*d, d))
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.query = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.output = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
                                 for _ in range(layers)])
        self.gate = nn.Sequential(nn.Linear(2*d, d), nn.Sigmoid())
        nn.init.constant_(self.gate[0].bias, -1)
        # Attributes of the camera writer that the trainer reads (no ray head here).
        self.depth_bins = 0
        self.depth_records = []
        self.record_depth = False

    def _children(self, onehot, ego_from_scene, points):
        """One-hot child histograms [T,L,8*C] and centre validity [T,L] (no gradient)."""
        t = ego_from_scene.shape[0]
        # Voxel centres at +/- 0.2 m (a quarter of the 0.8 m sub-cell) around the sub-cell centre.
        q = .25*torch.tensor(self.cell_size, device=points.device)/2
        axis = torch.tensor([-1., 1.], device=points.device)
        offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(8, 3)*q
        lo = torch.tensor(self.bounds[:3], device=points.device)
        span = torch.tensor(self.bounds[3:], device=points.device)-lo
        with torch.autocast(device_type=points.device.type, enabled=False):
            centre = transform_points(ego_from_scene, points)                       # T,L,3
            valid = ((centre >= lo) & (centre < lo+span)).all(-1)
            child = transform_points(ego_from_scene, (points[:, None]+offsets).reshape(-1, 3))
            grid = (2*(child-lo)/span-1).reshape(t, 1, 1, -1, 3)
            sampled = F.grid_sample(onehot, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return sampled.reshape(t, self.classes, -1, 8).permute(0, 2, 3, 1).reshape(t, -1, 8*self.classes), valid

    def _chunk(self, prior, points, onehot, ego_from_scene, fresh):
        """Write one chunk of sub-cells [L,d] from the label grids of the fresh input frames."""
        with torch.no_grad():
            hist, valid = self._children(onehot, ego_from_scene, points)
        valid = (valid & fresh[:, None]).t()                                             # L,T
        tokens = self.embed(hist).transpose(0, 1)                                          # L,T,d
        k = self.key(tokens)
        v = self.value(tokens)
        # Null evidence (zero key/value) keeps unobserved sub-cells finite.
        k = torch.cat([k, torch.zeros_like(k[:, :1])], 1)[:, None]
        v = torch.cat([v, torch.zeros_like(v[:, :1])], 1)[:, None]
        allowed = torch.cat([valid, torch.ones_like(valid[:, :1])], 1)[:, None, None]    # L,1,1,T+1
        candidate = prior
        for norm, qproj, out, ff in zip(self.norm, self.query, self.output, self.ff):
            q = qproj(norm(candidate))[:, None, None]
            evidence = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype),
                                                      attn_mask=allowed).reshape(prior.shape[0], self.dim)
            candidate = candidate+out(evidence)
            candidate = candidate+ff(candidate)
        seen = valid.any(1)
        gate = self.gate(torch.cat([prior, candidate.to(prior.dtype)], -1))*seen[:, None]
        return prior+gate*(candidate-prior), seen

    def forward(self, state, features, image_from_scene, times, image_hw, patch_grid, reference_points):
        """Same interface as CameraWriter.forward. ``features`` holds the label grids [1,T,1,X,Y,Z] and
        ``image_from_scene`` the ego_from_scene transform of each input frame (one "camera" per frame).
        Returns the posterior and the per-cell coverage."""
        if features.shape[0] != 1 or features.shape[2] != 1 or tuple(features.shape[3:]) != self.grid_shape:
            raise ValueError('Occupancy writer expects one scene and [1,T,1,X,Y,Z] label grids')
        if state.features.shape[-1] != self.packed_dim:
            raise ValueError('Occupancy writer needs the packed state')
        if hs.checks() and torch.any(times[:, -1] != state.time):
            raise ValueError('State must be at the block end')
        fresh = times >= state.last_observation_time[:, None]
        labels = features[0, :, 0].long().clamp_(0, self.classes-1)
        # [T,C,Z,Y,X] one-hot volume; grid_sample (x,y,z) indexes (W,H,D).
        onehot = F.one_hot(labels, self.classes).permute(0, 4, 3, 2, 1).float()
        fine = unpack_subcells(state.features)
        prior = fine.reshape(-1, self.dim)
        points = subcell_points(reference_points, self.cell_size).reshape(-1, 3).float()
        ego_from_scene = image_from_scene[0, :, 0].float()
        chunk = 8*self.query_chunk
        pieces, coverage = [], []
        for start in range(0, prior.shape[0], chunk):
            args = (prior[start:start+chunk], points[start:start+chunk], onehot, ego_from_scene, fresh[0])
            value, seen = (checkpoint(self._chunk, *args, use_reentrant=False)
                           if self.training and self.checkpoint_chunks else self._chunk(*args))
            pieces.append(value)
            coverage.append(seen)
        features = pack_subcells(torch.cat(pieces).reshape(fine.shape))
        seen = pack_subcells(torch.cat(coverage).reshape(fine.shape[:-1])[..., None]).any(-1)
        result = SceneField(features, state.world_from_ego, state.time,
                            torch.maximum(state.last_observation_time, times[:, -1]), state.updates+1)
        return result, seen
