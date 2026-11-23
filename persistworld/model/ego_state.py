"""Ablation ``abl_ego_state`` (Tab. 3, Ego-centric state): the world state on a lattice that follows the ego.

The model keeps its state on a lattice fixed in the scene (``Session.world_from_scene`` is the first ego pose
of the stream, so a cell keeps its address while the vehicle moves). In this ablation the lattice follows the
vehicle instead: after each capture group has been written, the whole stored state is resampled into the ego
frame of that group, ``world_from_scene`` becomes that ego pose, content outside the region around the ego is
dropped, and cells entering the region later are unwritten (they take the learned initial feature when read).

Every other component reads the lattice frame only through ``world_from_scene`` (the writer's projections, the
evolver's window and ego-relative positions, the shared context, the reader's resampling into the queried ego
frame), so they run unchanged on the moving lattice.

Resampling of the stored state (once per capture group):
* region: the evolver's window around the ego (``window_box`` of the identity pose, with the halo);
* features: trilinear interpolation on the 0.8 m sub-cell lattice (unpack, sample, repack). Source cells that
  were never written contribute the learned initial feature of their position relative to the new ego, i.e.
  what the evolver and the reader would read there; differentiable in the state and the initial field;
* time of the estimate, time of the last observation and the stored flag: from the source cell containing
  the new cell centre, so a cell stays stored only where its source was written.
Without ego motion (identical pose) the state is left untouched.
"""
import torch
from torch.nn import functional as F

from . import geometry
from . import host_sync as hs
from .memory import MemoryTile, WorldWindow, window_box
from .state import pack_subcells, subcell_points, unpack_subcells


def _region(model):
    """Integer cell box of the region around the ego, in the ego-aligned lattice (host values)."""
    return window_box(torch.eye(4, dtype=torch.float64), model.config.bounds, model.cell_size, model.config.halo)


def _sample_grid(old_from_new, points, src_origin, src_shape, size):
    """grid_sample coordinates (align_corners=False) of new-frame points in a source box of ``size`` cells."""
    q = geometry.transform_points(old_from_new, points.reshape(-1, 3)).reshape_as(points)
    lower = hs.upload(list(src_origin), q)*hs.const(list(size), q)
    span = hs.const(list(src_shape), q)*hs.const(list(size), q)
    return 2*(q-lower)/span-1


def resample(model, memory, old_world_from_scene, new_world_from_scene, default_time, dtype):
    """The stored state as a dense window of the new lattice."""
    cell = model.cell_size
    origin, shape = _region(model)
    device = old_world_from_scene.device
    # Source box (old lattice) covering the new region plus one interpolation cell; host geometry.
    old_from_new_host = torch.linalg.solve(hs.host(old_world_from_scene).double(),
                                           hs.host(new_world_from_scene).double())
    extent = [o*s for o, s in zip(origin, cell)]+[(o+n)*s for o, n, s in zip(origin, shape, cell)]
    src_origin, src_shape = window_box(old_from_new_host, extent, cell, 1)
    old_from_new = geometry.solve(old_world_from_scene, new_world_from_scene)
    new_from_old = geometry.inverse(old_from_new)
    # Source window; unwritten cells take the learned initial feature relative to the new ego frame.
    axes = [(torch.arange(n, device=device).float()+o+.5)*s for n, o, s in zip(src_shape, src_origin, cell)]
    src_points = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
    ego_points = geometry.transform_points(new_from_old, src_points.reshape(-1, 3)).reshape_as(src_points)
    low = hs.const(list(model.config.bounds[:3]), ego_points)
    span = hs.const(list(model.config.bounds[3:]), ego_points)-low
    initial = model.initial(2*(ego_points-low)/span-1)[None].to(dtype)   # keep the stored precision
    source = memory.read(src_origin, initial, default_time)
    with torch.autocast(device_type=device.type, enabled=False):
        # Features: trilinear on the 0.8 m sub-cell lattice.
        volume = unpack_subcells(source.features)
        fine = tuple(s/2 for s in cell)
        new_axes = [(torch.arange(n, device=device).float()+o+.5)*s for n, o, s in zip(shape, origin, cell)]
        coarse = torch.stack(torch.meshgrid(*new_axes, indexing='ij'), -1)
        grid = _sample_grid(old_from_new, subcell_points(coarse, cell), [2*o for o in src_origin],
                            [2*n for n in src_shape], fine)
        sampled = F.grid_sample(volume.permute(0, 4, 3, 2, 1).float(), grid.permute(2, 1, 0, 3)[None].float(),
                                mode='bilinear', padding_mode='border', align_corners=False)
        features = pack_subcells(sampled.permute(0, 4, 3, 2, 1)).to(dtype)
        # Time fields and the stored flag: the source cell containing the new cell centre.
        q = geometry.transform_points(old_from_new, coarse.reshape(-1, 3))
        index = torch.floor(q/hs.const(list(cell), q)).long()-hs.upload(list(src_origin), q, dtype=torch.long)
        for axis, n in enumerate(src_shape):
            index[:, axis] = index[:, axis].clamp(0, n-1)
        flat = (index[:, 0]*src_shape[1]+index[:, 1])*src_shape[2]+index[:, 2]
        pick = lambda x: x.reshape(-1).index_select(0, flat).reshape(1, *shape)
        return WorldWindow(features, tuple(origin), tuple(cell), pick(source.state_time),
                           pick(source.observed_time), pick(source.valid))


def recenter(model, session, world_from_ego, dtype):
    """Move the lattice of ``session`` to ``world_from_ego`` [1,4,4], the pose just written."""
    new = hs.view(world_from_ego, lambda x: x[0].detach().clone())
    memory = session.memory
    if torch.equal(hs.host(session.world_from_scene).double(), hs.host(new).double()):
        return   # no ego motion: the lattice already is the ego frame
    window = resample(model, memory, session.world_from_scene, new, session.last_time, dtype)
    memory.tiles = {}   # content outside the region is dropped (a new dict: forks keep theirs)
    memory.write(window)
    # write() marks every covered cell as stored; keep the flag of the source cell instead.
    for key, local, part in memory._overlap(window.origin, window.shape):
        tile = memory.tiles[key]
        valid = torch.zeros_like(tile.valid)
        valid[part] = window.valid[0][local]
        memory.tiles[key] = MemoryTile(tile.features, tile.state_time, tile.observed_time, valid)
    session.world_from_scene = new
