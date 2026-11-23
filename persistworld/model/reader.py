"""Reader: decodes occupancy from the packed state.

The reader unpacks the sub-cells and applies a local residual block (a depth-wise 3^3 convolution
and point-wise mixing), followed by a linear layer that predicts class logits for the 2x2x2 voxels
of 0.4 m inside each sub-cell. The logits are resampled from the fixed world lattice into the
queried ego frame.
"""
import torch
from torch import nn
from torch.nn import functional as F

from . import geometry
from . import host_sync as hs
from .state import reference_grid, unpack_subcells


class Reader(nn.Module):
    """Local decoder from packed cell features to the class logits of their 0.4 m voxels."""
    def __init__(self, packed_dim, classes=18):
        super().__init__()
        if packed_dim < 16 or packed_dim % 8:
            raise ValueError('Expected eight packed subcells')
        dim = packed_dim // 8
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.mix = nn.Conv3d(dim, dim, 1)
        self.out = nn.Linear(dim, 8*classes)
        self.planned_sampling = True   # execution switch: Triton trilinear sampling on CUDA without gradient

    def forward(self, packed_features):
        """[B,X,Y,Z,8C] packed cells -> [B,4X,4Y,4Z,classes] logits of the 0.4 m voxels."""
        x = unpack_subcells(packed_features)
        local = self.norm(x).permute(0, 4, 3, 2, 1)
        local = self.mix(F.gelu(self.depthwise(local)))
        return unpack_subcells(self.out(x + local.permute(0, 4, 3, 2, 1)))


def read_world_query(reader, query, output_shape, bounds):
    """Decode ``query`` (PackedWorldQuery) and resample the logits onto the ego occupancy grid."""
    if query.features.shape[0] != 1:
        raise ValueError('World readout requests retain one scene origin each')
    logits = reader(query.features)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        xyz = reference_grid(output_shape, bounds, device=logits.device)
        points = geometry.transform_points(query.scene_from_ego, xyz.reshape(-1, 3))
        lower = hs.upload(list(query.origin), xyz) * hs.const(list(query.cell_size), xyz)
        span = hs.const(list(query.features.shape[1:4]), xyz) * hs.const(list(query.cell_size), xyz)
        grid = (2*(points-lower)/span-1).reshape(*output_shape, 3)
        if getattr(reader, 'planned_sampling', False) and logits.is_cuda and not grid.requires_grad:
            from ..kernels.planned_sampling import planned_grid_sample
            return planned_grid_sample(logits, grid)
        grid = grid.permute(2, 1, 0, 3)[None]
        sampled = F.grid_sample(logits.permute(0, 4, 3, 2, 1).float(), grid.float(), align_corners=False)
    return sampled.permute(0, 4, 3, 2, 1).to(logits.dtype)
