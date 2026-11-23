"""Execution-only helpers for a forward pass without host/device round trips.

With ``ENABLED`` (execution option ``sync_free_forward``) the forward never waits for the GPU:
small constants are uploaded once from pinned memory and cached, per-query values are uploaded
asynchronously, host-side decisions read CPU mirrors of the (CPU-born) input tensors, and
device-side input validation is skipped because the same inputs are validated on the host.
Disabled, every helper reproduces the plain synchronous call.
"""
import torch

ENABLED = False
_constants = {}


def const(values, like=None, device=None, dtype=None):
    """Like ``like.new_tensor(values)`` / ``torch.tensor(values, device=...)``; read-only."""
    device = like.device if like is not None else torch.device(device)
    dtype = dtype if dtype is not None else (like.dtype if like is not None else torch.float32)
    if not (ENABLED and device.type == 'cuda'):
        return torch.tensor(values, device=device, dtype=dtype)
    key = (repr(values), device, dtype)
    value = _constants.get(key)
    if value is None:
        value = torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)
        _constants[key] = value
    return value


def upload(values, like=None, device=None, dtype=None):
    """Per-query values (not cached), asynchronous from pinned memory when enabled."""
    device = like.device if like is not None else torch.device(device)
    dtype = dtype if dtype is not None else (like.dtype if like is not None else torch.float32)
    if not (ENABLED and device.type == 'cuda'):
        return torch.tensor(values, device=device, dtype=dtype)
    return torch.tensor(values, dtype=dtype).pin_memory().to(device, non_blocking=True)


def to_device(tensor, device):
    """CPU tensor -> device; asynchronous (pinned) when enabled. Keeps a CPU mirror."""
    if not (ENABLED and torch.device(device).type == 'cuda' and tensor.device.type == 'cpu'):
        return tensor.to(device)
    source = tensor if tensor.is_pinned() else tensor.pin_memory()
    return mirror(source.to(device, non_blocking=True), tensor)


def mirror(value, host_value):
    """Attach the CPU value of a device tensor (no copy)."""
    if ENABLED and host_value is not None:
        value._host = host_value
    return value


def host(tensor):
    """CPU value of ``tensor``: its mirror when available, else a (synchronizing) copy."""
    value = getattr(tensor, '_host', None) if ENABLED else None
    return tensor.detach().cpu() if value is None else value


def checks():
    """Whether device-side input validation runs (always, unless sync-free)."""
    return not ENABLED


def view(tensor, fn):
    """``fn(tensor)`` with the mirror carried through (slicing, detach, indexing by Python ints)."""
    out = fn(tensor)
    value = getattr(tensor, '_host', None) if ENABLED else None
    return out if value is None else mirror(out, fn(value))


def take(tensor, dim, keep):
    """``tensor.index_select(dim, keep)`` for a Python list; a slice when contiguous."""
    keep = list(keep)
    if keep and keep == list(range(keep[0], keep[-1]+1)):
        return view(tensor, lambda x: x.narrow(dim, keep[0], len(keep)))
    out = tensor.index_select(dim, const(keep, device=tensor.device, dtype=torch.long))
    value = getattr(tensor, '_host', None) if ENABLED else None
    return out if value is None else mirror(out, value.index_select(dim, torch.tensor(keep)))


def cat(tensors, dim):
    """``torch.cat`` that keeps the CPU mirror when every input has one."""
    out = torch.cat(tensors, dim)
    values = [getattr(t, '_host', None) for t in tensors] if ENABLED else [None]
    return out if any(v is None for v in values) else mirror(out, torch.cat(values, dim))
