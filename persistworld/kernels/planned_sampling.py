"""Constant-coordinate trilinear readout with shared FP32 interpolation plans.

Matches 5-D grid_sample, zero padding and align_corners=False. Geometry keeps
the reference FP32 normalization and unnormalization. Only the logits receive
gradients; callers with differentiable coordinates must use grid_sample.
Indices/weights are saved once per output voxel and reused across channels
and backward. The accumulation buffer remains FP32, including BF16 inputs.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['X', 'Y', 'Z'])
def _plan(Grid, Indices, Weights, N: tl.constexpr,
          X, Y, Z, BLOCK: tl.constexpr):
    p = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    x = ((tl.load(Grid+3*p, p < N, 0)+1)*X-1)*.5
    y = ((tl.load(Grid+3*p+1, p < N, 0)+1)*Y-1)*.5
    z = ((tl.load(Grid+3*p+2, p < N, 0)+1)*Z-1)*.5
    x0 = tl.floor(x).to(tl.int32)
    y0 = tl.floor(y).to(tl.int32)
    z0 = tl.floor(z).to(tl.int32)
    for corner in tl.static_range(8):
        ix = x0+(corner & 1)
        iy = y0+((corner>>1) & 1)
        iz = z0+((corner>>2) & 1)
        wx = (x-x0) if corner & 1 else (x0+1-x)
        wy = (y-y0) if corner & 2 else (y0+1-y)
        wz = (z-z0) if corner & 4 else (z0+1-z)
        valid = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        idx = tl.where(valid, (ix*Y+iy)*Z+iz, 0)
        weight = tl.where(valid, wx*wy*wz, 0.)
        tl.store(Indices+corner*N+p, idx, p < N)
        tl.store(Weights+corner*N+p, weight, p < N)


@triton.jit
def _gather(Value, Indices, Weights, Out, N: tl.constexpr, C: tl.constexpr,
            P: tl.constexpr, K: tl.constexpr):
    p = tl.program_id(0)*P+tl.arange(0, P)
    c = tl.arange(0, K)
    result = tl.full((P, K), 0, tl.float32)
    for corner in tl.static_range(8):
        idx = tl.load(Indices+corner*N+p, p < N, 0)
        w = tl.load(Weights+corner*N+p, p < N, 0)
        v = tl.load(Value+idx[:, None]*C+c[None, :], (p[:, None] < N) & (c[None, :] < C), 0).to(tl.float32)
        result = result+v*w[:, None]
    tl.store(Out+p[:, None]*C+c[None, :], result, (p[:, None] < N) & (c[None, :] < C))


@triton.jit
def _scatter(Upstream, Indices, Weights, Grad, N: tl.constexpr, C: tl.constexpr,
             P: tl.constexpr, K: tl.constexpr):
    p = tl.program_id(0)*P+tl.arange(0, P)
    c = tl.arange(0, K)
    g = tl.load(Upstream+p[:, None]*C+c[None, :], (p[:, None] < N) & (c[None, :] < C), 0).to(tl.float32)
    for corner in tl.static_range(8):
        idx = tl.load(Indices+corner*N+p, p < N, 0)
        w = tl.load(Weights+corner*N+p, p < N, 0)
        tl.atomic_add(Grad+idx[:, None]*C+c[None, :], g*w[:, None],
                      (p[:, None] < N) & (c[None, :] < C) & (w[:, None] != 0), sem='relaxed')


class _PlannedSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, grid):
        _, x, y, z, c = value.shape
        n = grid.numel()//3
        indices = torch.empty((8, n), device=value.device, dtype=torch.int32)
        weights = torch.empty((8, n), device=value.device, dtype=torch.float32)
        out = torch.empty((1, *grid.shape[:-1], c), device=value.device, dtype=value.dtype)
        # Native CUDA grid_sample uses fused multiply-add in coordinate
        # unnormalization and weighted accumulation; match that rounding.
        _plan[(triton.cdiv(n, 256),)](grid, indices, weights, n, x, y, z, 256, enable_fp_fusion=True)
        _gather[(triton.cdiv(n, 32),)](value, indices, weights, out, n, c, 32,
                                    triton.next_power_of_2(c), enable_fp_fusion=True)
        ctx.save_for_backward(indices, weights)
        ctx.input_shape = value.shape
        ctx.dtype = value.dtype
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        indices, weights = ctx.saved_tensors
        n = indices.shape[1]
        c = ctx.input_shape[-1]
        grad = torch.zeros(ctx.input_shape, device=upstream.device, dtype=torch.float32)
        _scatter[(triton.cdiv(n, 32),)](upstream.contiguous(), indices, weights, grad, n, c, 32,
                                     triton.next_power_of_2(c), enable_fp_fusion=False)
        return grad.to(ctx.dtype), None


def planned_grid_sample(value, grid):
    """value [1,X,Y,Z,C], grid [Xo,Yo,Zo,3] normalized (x, y, z) -> [1,Xo,Yo,Zo,C]; grid_sample semantics."""
    if value.ndim != 5 or value.shape[0] != 1 or grid.ndim != 4 or grid.shape[-1] != 3:
        raise ValueError('Expected one [X,Y,Z,C] volume and [Xout,Yout,Zout,3] grid')
    if grid.requires_grad:
        raise ValueError('Planned readout requires constant coordinates')
    if not value.is_cuda or grid.device != value.device or grid.dtype != torch.float32:
        raise ValueError('Planned readout requires same-device CUDA FP32 coordinates')
    return _PlannedSampling.apply(value.contiguous(), grid.contiguous())
