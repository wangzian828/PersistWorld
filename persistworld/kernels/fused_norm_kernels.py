"""Triton kernels of ``fused_norm``: the two row-wise layer norms of LN((1 + s) * LN(x) + t) and their
analytic backward, one program per row.

LayerNorm formula/reference: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
Unlike the tutorial there is no reduction across rows: the modulation differs for every cell, so its
gradients are per element.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, S, T, Y, ST, XS: tl.constexpr, SS: tl.constexpr, TS: tl.constexpr,
             N: tl.constexpr, E1: tl.constexpr, E2: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    valid = c < N
    x = tl.load(X+row*XS+c, valid, 0).to(tl.float32)
    s = tl.load(S+row*SS+c, valid, 0)
    t = tl.load(T+row*TS+c, valid, 0).to(tl.float32)
    a = (s.to(tl.float32)+1.).to(s.dtype).to(tl.float32)
    m = tl.sum(x, 0)/N
    centered = tl.where(valid, x-m, 0.)
    r = tl.rsqrt(tl.sum(centered*centered, 0)/N+E1)
    z = tl.where(valid, centered*r*a+t, 0.)
    m2 = tl.sum(z, 0)/N
    centered2 = tl.where(valid, z-m2, 0.)
    r2 = tl.rsqrt(tl.sum(centered2*centered2, 0)/N+E2)
    tl.store(Y+row*N+c, centered2*r2, valid)
    tl.store(ST+row*4, m)
    tl.store(ST+row*4+1, r)
    tl.store(ST+row*4+2, m2)
    tl.store(ST+row*4+3, r2)


@triton.jit
def _backward(X, S, T, G, ST, DX, DS, DT, XS: tl.constexpr, SS: tl.constexpr, TS: tl.constexpr,
              N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    valid = c < N
    x = tl.load(X+row*XS+c, valid, 0).to(tl.float32)
    s = tl.load(S+row*SS+c, valid, 0)
    t = tl.load(T+row*TS+c, valid, 0).to(tl.float32)
    g = tl.load(G+row*N+c, valid, 0).to(tl.float32)
    m = tl.load(ST+row*4)
    r = tl.load(ST+row*4+1)
    m2 = tl.load(ST+row*4+2)
    r2 = tl.load(ST+row*4+3)
    a = (s.to(tl.float32)+1.).to(s.dtype).to(tl.float32)
    n = tl.where(valid, (x-m)*r, 0.)
    n2 = tl.where(valid, (n*a+t-m2)*r2, 0.)
    dz = tl.where(valid, (g-tl.sum(g, 0)/N-n2*tl.sum(g*n2, 0)/N)*r2, 0.)
    dn = dz*a
    dx = (dn-tl.sum(dn, 0)/N-n*tl.sum(dn*n, 0)/N)*r
    tl.store(DX+row*N+c, dx, valid)
    tl.store(DS+row*N+c, dz*n, valid)
    tl.store(DT+row*N+c, dz, valid)


def rows(t):
    """[..., N] -> [rows, N] view with unit column stride (copied only when needed)."""
    value = t.reshape(-1, t.shape[-1])
    # The normal six-way condition split has row stride 6D and column stride 1;
    # preserve it, avoiding copies of the three modulation tensors.
    return value if value.stride(1) == 1 else value.contiguous()


def launch_forward(x, s, t, eps1, eps2):
    a, b, c = rows(x), rows(s), rows(t)
    m, n = a.shape
    y = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    stats = torch.empty((m, 4), device=x.device, dtype=torch.float32)
    _forward[(m,)](a, b, c, y, stats, a.stride(0), b.stride(0), c.stride(0), n, eps1, eps2,
                   triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    return y, stats


def launch_backward(x, s, t, stats, g):
    a, b, c = rows(x), rows(s), rows(t)
    m, n = a.shape
    dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    ds = torch.empty(s.shape, device=s.device, dtype=s.dtype)
    dt = torch.empty(t.shape, device=t.device, dtype=t.dtype)
    _backward[(m,)](a, b, c, g.contiguous(), stats, dx, ds, dt, a.stride(0), b.stride(0), c.stride(0), n,
                    triton.next_power_of_2(n), num_warps=4, enable_fp_fusion=False)
    return dx, ds, dt
