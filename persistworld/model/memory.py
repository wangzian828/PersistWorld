"""Scene memory: the world state stored in fixed-address tiles of the scene lattice.

The state is stored only where it has been written. A read returns a window of the lattice
around the ego; cells without a stored estimate take the learned initial feature. Reads and
writes gather/scatter all overlapping tiles at once and are differentiable. Writes are
copy-on-write, so earlier autograd graphs and session snapshots stay valid.
"""
from dataclasses import dataclass
from functools import lru_cache
import itertools

import numpy as np
import torch
from torch.nn import functional as F

from . import host_sync as hs
from .geometry import transform_points


@dataclass
class WorldWindow:
    """A dense box of the scene lattice: features and per-cell times at integer cell ``origin``."""
    features: torch.Tensor                 # [1,X,Y,Z,D]
    origin: tuple                          # integer cell coordinates
    cell_size: tuple
    state_time: torch.Tensor               # [1,X,Y,Z] time of the estimate
    observed_time: torch.Tensor            # [1,X,Y,Z] time of the last observation
    valid: torch.Tensor                    # a stored estimate exists

    @property
    def shape(self):
        return tuple(self.features.shape[1:4])

    def points(self):
        """Cell centres [X,Y,Z,3] in the scene frame."""
        axes = [(torch.arange(n, device=self.features.device, dtype=torch.float32)+o+.5)*s
                for n, o, s in zip(self.shape, self.origin, self.cell_size)]
        return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)

    def replace(self, features, **kwargs):
        """Copy with new ``features`` and optionally other fields."""
        values = dict(origin=self.origin, cell_size=self.cell_size, state_time=self.state_time,
                      observed_time=self.observed_time, valid=self.valid)
        values.update(kwargs)
        return WorldWindow(features, **values)


@dataclass
class MemoryTile:
    """One stored tile of the scene memory."""
    features: torch.Tensor                 # [X,Y,Z,D], no batch
    state_time: torch.Tensor
    observed_time: torch.Tensor
    valid: torch.Tensor

    def to(self, device, detach=False):
        def move(x):
            return (x.detach() if detach else x).to(device)
        return MemoryTile(*(move(getattr(self, k)) for k in ('features', 'state_time', 'observed_time', 'valid')))


def window_box(scene_from_ego, bounds, cell_size, halo=1):
    """Integer window covering the eight corners of the rotated ego volume, plus an interpolation halo."""
    corners = torch.tensor(list(itertools.product(*[(bounds[i], bounds[i+3]) for i in range(3)])),
                           device=scene_from_ego.device, dtype=torch.float32)
    xyz = transform_points(scene_from_ego, corners)
    size = xyz.new_tensor(cell_size)
    lo = torch.floor(xyz.amin(0)/size).long()-halo
    hi = torch.ceil(xyz.amax(0)/size).long()+halo
    return tuple(lo.tolist()), tuple((hi-lo).tolist())


class SceneMemory:
    """Tiles of the scene lattice keyed by integer tile coordinates; ``revision`` counts the changes."""
    def __init__(self, dim, cell_size=(1.6, 1.6, 1.6), tile_shape=(10, 10, 4)):
        if dim <= 0 or any(s <= 0 for s in (*cell_size, *tile_shape)):
            raise ValueError('Positive memory dimensions required')
        self.dim = dim
        self.cell_size = tuple(cell_size)
        self.tile_shape = tuple(tile_shape)
        self.tiles = {}
        self.revision = 0

    def _overlap(self, origin, shape):
        """(tile key, window slices, tile slices) of every tile a window overlaps."""
        ranges = [range(o//s, (o+n-1)//s+1) for o, n, s in zip(origin, shape, self.tile_shape)]
        for key in itertools.product(*ranges):
            base = tuple(k*s for k, s in zip(key, self.tile_shape))
            lo = tuple(max(o, b) for o, b in zip(origin, base))
            hi = tuple(min(o+n, b+s) for o, n, b, s in zip(origin, shape, base, self.tile_shape))
            local = tuple(slice(l-o, h-o) for l, h, o in zip(lo, hi, origin))
            tile = tuple(slice(l-b, h-b) for l, h, b in zip(lo, hi, base))
            yield key, local, tile

    def read(self, origin, initial, default_time):
        """Window at ``origin`` of the shape of ``initial`` [1,X,Y,Z,D], which fills the unstored cells."""
        if initial.ndim != 5 or initial.shape[0] != 1 or initial.shape[-1] != self.dim:
            raise ValueError('One scene per memory instance; expected [1,X,Y,Z,D]')
        return read_window(self, origin, initial, default_time)

    def write(self, window, observed_time=None):
        """Store ``window`` (copy-on-write); returns the keys of the tiles written."""
        if window.cell_size != self.cell_size:
            raise ValueError('Mismatched lattice')
        return write_window(self, window, observed_time)

    def fork(self):
        """Cheap read snapshot: writes replace tiles instead of mutating them."""
        bank = SceneMemory(self.dim, self.cell_size, self.tile_shape)
        bank.tiles = self.tiles.copy()
        bank.revision = self.revision
        return bank


@lru_cache(maxsize=64)
def address_map(origin, shape, tile_shape, keys):
    """Flat index of every window cell into the stacked tiles ``keys`` (-1: no stored tile)."""
    ids = np.full(shape, -1, dtype=np.int64)
    local_ids = np.arange(int(np.prod(tile_shape)), dtype=np.int64).reshape(tile_shape)
    volume = int(np.prod(tile_shape))
    for index, key in enumerate(keys):
        base = tuple(k*s for k, s in zip(key, tile_shape))
        lo = tuple(max(o, b) for o, b in zip(origin, base))
        hi = tuple(min(o+n, b+s) for o, n, b, s in zip(origin, shape, base, tile_shape))
        local = tuple(slice(l-o, h-o) for l, h, o in zip(lo, hi, origin))
        part = tuple(slice(l-b, h-b) for l, h, b in zip(lo, hi, base))
        ids[local] = index*volume+local_ids[part]
    return torch.from_numpy(ids.reshape(-1))


def read_window(bank, origin, initial, default_time):
    """Gather the stored cells of a window in one pass; unstored cells take ``initial`` and ``default_time``."""
    shape = tuple(initial.shape[1:4])
    device = initial.device
    keys = tuple(key for key, _, _ in bank._overlap(origin, shape) if key in bank.tiles)
    times = torch.full(shape, float(default_time), device=device, dtype=torch.float32)
    observed = torch.full_like(times, -torch.inf)
    if not keys:
        return WorldWindow(initial, tuple(origin), bank.cell_size, times[None], observed[None],
                           torch.zeros((1, *shape), device=device, dtype=torch.bool))
    index = hs.to_device(address_map(tuple(origin), shape, bank.tile_shape, keys), device)
    selected = index.clamp_min(0)
    tiles = [bank.tiles[key].to(device) for key in keys]

    def gather(name, feature=False):
        values = torch.stack([getattr(t, name) for t in tiles])
        if feature:
            return values.reshape(-1, bank.dim).index_select(0, selected).reshape(*shape, bank.dim)
        return values.reshape(-1).index_select(0, selected).reshape(shape)

    valid = gather('valid') & (index >= 0).reshape(shape)
    values = torch.where(valid[..., None], gather('features', True).to(initial.dtype), initial[0])
    times = torch.where(valid, gather('state_time'), times)
    observed = torch.where(valid, gather('observed_time'), observed)
    return WorldWindow(values[None], tuple(origin), bank.cell_size, times[None], observed[None], valid[None])


def write_window(bank, window, observed_time=None):
    """Pack all touched tiles once, preserving copy-on-write and the old tile contents outside the window."""
    overlaps = list(bank._overlap(window.origin, window.shape))
    keys = [row[0] for row in overlaps]
    tile = bank.tile_shape
    shape = window.shape
    dim = bank.dim
    device = window.features.device
    low = tuple(o % s for o, s in zip(window.origin, tile))
    counts = tuple((l+n+s-1)//s for l, n, s in zip(low, shape, tile))
    high = tuple(c*s-l-n for c, s, l, n in zip(counts, tile, low, shape))
    pad = tuple(v for axis in (2, 1, 0) for v in (low[axis], high[axis]))

    def pack(value, feature=False, fill=0):
        padded = F.pad(value, (0, 0, *pad) if feature else pad, value=fill)
        order = (0, 2, 4, 1, 3, 5, 6) if feature else (0, 2, 4, 1, 3, 5)
        dimensions = tuple(v for c, s in zip(counts, tile) for v in (c, s))
        suffix = (dim,) if feature else ()
        return padded.reshape(*dimensions, *suffix).permute(order).reshape(len(keys), *tile, *suffix)

    inside = pack(torch.ones(shape, device=device, dtype=torch.bool))
    new = dict(features=pack(window.features[0], True), state_time=pack(window.state_time[0]),
               observed_time=pack((window.observed_time if observed_time is None else observed_time)[0]))
    defaults = dict(features=torch.zeros((*tile, dim), device=device, dtype=window.features.dtype),
                    state_time=torch.zeros(tile, device=device),
                    observed_time=torch.full(tile, -torch.inf, device=device),
                    valid=torch.zeros(tile, device=device, dtype=torch.bool))
    previous = [bank.tiles[key].to(device) if key in bank.tiles else None for key in keys]
    result = {}
    for name in ('features', 'state_time', 'observed_time', 'valid'):
        if any(t is not None for t in previous):
            old = torch.stack([getattr(t, name) if t is not None else defaults[name] for t in previous])
            if name == 'features' and old.dtype != window.features.dtype:
                raise ValueError('Packed writes require consistent feature dtype')
            value = (inside | old if name == 'valid'
                     else torch.where(inside[..., None] if name == 'features' else inside, new[name], old))
        else:
            value = (inside if name == 'valid'
                     else torch.where(inside[..., None] if name == 'features' else inside, new[name], defaults[name]))
        # One UnbindBackward gathers the tile gradients.
        result[name] = value.unbind(0)
    for index, key in enumerate(keys):
        bank.tiles[key] = MemoryTile(*(result[name][index]
                                       for name in ('features', 'state_time', 'observed_time', 'valid')))
    bank.revision += 1
    return set(keys)
