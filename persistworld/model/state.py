"""World-state containers and the packed sub-cell layout.

The state lives on a lattice of 1.6 m cells fixed in the scene. Each cell feature packs the
features of its 2x2x2 sub-cells of 0.8 m; the channel order is (subcell_x, subcell_y, subcell_z,
feature). Packing is a permutation, never an interpolation. The evolver attends over the packed
lattice; the writer and the reader operate on the unpacked sub-cells.
"""
from dataclasses import dataclass

import torch

from . import host_sync as hs


@dataclass
class SceneField:
    """Packed features [B,X,Y,Z,D] with their pose, time and last observation time."""
    features: torch.Tensor
    world_from_ego: torch.Tensor
    time: torch.Tensor
    last_observation_time: torch.Tensor
    updates: int = 0


@dataclass(kw_only=True)
class PackedWorldQuery(SceneField):
    """Reader request that keeps the fixed-world addresses until after local decoding."""
    origin: tuple
    cell_size: tuple
    scene_from_ego: torch.Tensor


def reference_grid(shape, bounds, *, device=None, dtype=torch.float32):
    """Voxel centres [X,Y,Z,3] of a ``shape`` grid spanning ``bounds`` (xmin, ymin, zmin, xmax, ymax, zmax)."""
    axes = [torch.linspace(bounds[i], bounds[i+3], shape[i]+1, device=device, dtype=dtype) for i in range(3)]
    axes = [(a[:-1]+a[1:])/2 for a in axes]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)


def unpack_subcells(value):
    """[B,X,Y,Z,8C] packed cells -> [B,2X,2Y,2Z,C] sub-cells."""
    if value.ndim != 5 or value.shape[-1] % 8:
        raise ValueError('Packed state needs [B,X,Y,Z,8*C]')
    b, x, y, z, channels = value.shape
    return (value.reshape(b, x, y, z, 2, 2, 2, channels // 8)
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .reshape(b, 2*x, 2*y, 2*z, channels // 8))


def pack_subcells(value):
    """[B,2X,2Y,2Z,C] sub-cells -> [B,X,Y,Z,8C] packed cells."""
    if value.ndim != 5 or any(n % 2 for n in value.shape[1:4]):
        raise ValueError('Subcell spatial dimensions must be even')
    b, x, y, z, channels = value.shape
    return (value.reshape(b, x//2, 2, y//2, 2, z//2, 2, channels)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(b, x//2, y//2, z//2, 8*channels))


def subcell_points(coarse_points, cell_size):
    """Centres of the eight sub-cells, at +/- one quarter of each cell edge."""
    if coarse_points.ndim not in (4, 5) or coarse_points.shape[-1] != 3:
        raise ValueError('Expected spatial coarse centres, optionally batched')
    batched = coarse_points.ndim == 5
    points = coarse_points if batched else coarse_points[None]
    with torch.autocast(device_type=points.device.type, enabled=False):
        axis = hs.const([-.25, .25], device=points.device, dtype=torch.float32)
        offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1)
        offsets = offsets * hs.const(list(cell_size), offsets)
        b, x, y, z, _ = points.shape
        children = points.float()[..., None, None, None, :] + offsets
        children = (children.permute(0, 1, 4, 2, 5, 3, 6, 7)
                    .reshape(b, 2*x, 2*y, 2*z, 3))
    return children if batched else children[0]
