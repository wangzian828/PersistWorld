"""Fused Triton kernels for the camera writer's evidence path (execution only, same function).

``sample_tokens``: bilinear sampling of the packed key/value maps (zero padding, align_corners=False,
as F.grid_sample) written straight into the token-major [L, I*P, D] layout the writer attends over
(P = 1 sample per image), with the per-image camera tag added to the keys.

``attend``: one query per row against S keys plus the implicit null token (zero key, zero value,
its own bias), single head. Scores, softmax and accumulation are FP32; gradients for q, K, V,
bias and the null bias.
"""
import math

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


# ------------------------------------------------------------------ sampling

@triton.jit
def _corner_setup(Grid, i, l, lm, pair, L, H, W, P: tl.constexpr):
    g = Grid + ((i*L + l)*P + pair)*2
    gx = tl.load(g, lm, 0.)
    gy = tl.load(g + 1, lm, 0.)
    x = ((gx + 1)*W - 1)*.5
    y = ((gy + 1)*H - 1)*.5
    # Far/NaN coordinates only need to stay invalid (all corners outside).
    x = tl.where(x == x, tl.minimum(tl.maximum(x, -4.), W + 4.), -4.)
    y = tl.where(y == y, tl.minimum(tl.maximum(y, -4.), H + 4.), -4.)
    x0 = tl.floor(x)
    y0 = tl.floor(y)
    return x - x0, y - y0, x0.to(tl.int32), y0.to(tl.int32)


@triton.jit
def _sample_fwd(Maps, Grid, Tags, K, V, L, I, H, W, D: tl.constexpr, DP: tl.constexpr,
                BL: tl.constexpr, HAS_TAGS: tl.constexpr, P: tl.constexpr):
    i = tl.program_id(1)
    l = tl.program_id(0)*BL + tl.arange(0, BL)
    lm = l < L
    c = tl.arange(0, DP)
    cm = c < D
    for pair in tl.static_range(P):
        fx, fy, x0, y0 = _corner_setup(Grid, i, l, lm, pair, L, H, W, P)
        acc_k = tl.zeros((BL, DP), tl.float32)
        acc_v = tl.zeros((BL, DP), tl.float32)
        for corner in tl.static_range(4):
            cx = x0 + (corner & 1)
            cy = y0 + (corner >> 1)
            wx = fx if corner & 1 else 1 - fx
            wy = fy if corner & 2 else 1 - fy
            ok = lm & (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            base = ((i*H + tl.where(ok, cy, 0))*W + tl.where(ok, cx, 0))*(2*D)
            m = ok[:, None] & cm[None, :]
            w = (wx*wy)[:, None]
            acc_k += w*tl.load(Maps + base[:, None] + c[None, :], m, 0.).to(tl.float32)
            acc_v += w*tl.load(Maps + base[:, None] + D + c[None, :], m, 0.).to(tl.float32)
        if HAS_TAGS:
            acc_k += tl.load(Tags + i*D + c, cm, 0.).to(tl.float32)[None, :]
        out = (l[:, None]*(I*P) + i*P + pair)*D + c[None, :]
        om = lm[:, None] & cm[None, :]
        tl.store(K + out, acc_k.to(K.dtype.element_ty), om)
        tl.store(V + out, acc_v.to(V.dtype.element_ty), om)


@triton.jit
def _sample_bwd(Maps, Grid, DK, DV, DMaps, DGrid, L, I, H, W, D: tl.constexpr, DP: tl.constexpr,
                BL: tl.constexpr, P: tl.constexpr):
    i = tl.program_id(1)
    l = tl.program_id(0)*BL + tl.arange(0, BL)
    lm = l < L
    c = tl.arange(0, DP)
    cm = c < D
    for pair in tl.static_range(P):
        fx, fy, x0, y0 = _corner_setup(Grid, i, l, lm, pair, L, H, W, P)
        tok = (l[:, None]*(I*P) + i*P + pair)*D + c[None, :]
        tm = lm[:, None] & cm[None, :]
        gk = tl.load(DK + tok, tm, 0.).to(tl.float32)
        gv = tl.load(DV + tok, tm, 0.).to(tl.float32)
        grad_x = tl.zeros((BL,), tl.float32)
        grad_y = tl.zeros((BL,), tl.float32)
        for corner in tl.static_range(4):
            cx = x0 + (corner & 1)
            cy = y0 + (corner >> 1)
            wx = fx if corner & 1 else 1 - fx
            wy = fy if corner & 2 else 1 - fy
            ok = lm & (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
            base = ((i*H + tl.where(ok, cy, 0))*W + tl.where(ok, cx, 0))*(2*D)
            m = ok[:, None] & cm[None, :]
            w = (wx*wy)[:, None]
            tl.atomic_add(DMaps + base[:, None] + c[None, :], gk*w, m, sem='relaxed')
            tl.atomic_add(DMaps + base[:, None] + D + c[None, :], gv*w, m, sem='relaxed')
            dot = tl.sum(gk*tl.load(Maps + base[:, None] + c[None, :], m, 0.).to(tl.float32)
                         + gv*tl.load(Maps + base[:, None] + D + c[None, :], m, 0.).to(tl.float32), 1)
            grad_x += dot*(wy if corner & 1 else -wy)
            grad_y += dot*(wx if corner & 2 else -wx)
        g = DGrid + ((i*L + l)*P + pair)*2
        tl.store(g, grad_x*W*.5, lm)
        tl.store(g + 1, grad_y*H*.5, lm)


class _SampleTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, maps, grid, tags, out_dtype):
        i, h, w, c2 = maps.shape
        d = c2//2
        l, p = grid.shape[1], grid.shape[2]
        if grid.shape != (i, l, p, 2) or p not in (1, 2):
            raise ValueError('Expected grid [I,L,P,2] with P in {1,2}')
        maps = maps.contiguous()
        grid = grid.contiguous().float()
        k = torch.empty((l, i*p, d), device=maps.device, dtype=out_dtype)
        v = torch.empty_like(k)
        has_tags = tags is not None
        t = tags.contiguous() if has_tags else maps
        bl = 32
        _sample_fwd[(triton.cdiv(l, bl), i)](maps, grid, t, k, v, l, i, h, w, d,
                                            triton.next_power_of_2(d), bl, has_tags, p)
        ctx.save_for_backward(maps, grid)
        ctx.has_tags = has_tags
        ctx.tags_dtype = tags.dtype if has_tags else None
        return k, v

    @staticmethod
    @once_differentiable
    def backward(ctx, gk, gv):
        maps, grid = ctx.saved_tensors
        i, h, w, c2 = maps.shape
        d = c2//2
        l, p = grid.shape[1], grid.shape[2]
        gk = torch.zeros((l, i*p, d), device=maps.device, dtype=torch.float32) if gk is None else gk.contiguous()
        gv = torch.zeros_like(gk) if gv is None else gv.contiguous()
        dmaps = torch.zeros(maps.shape, device=maps.device, dtype=torch.float32)
        dgrid = torch.empty_like(grid)
        # Map-gradient atomics dominate; this block shape halves them on A100.
        bl = 8
        _sample_bwd[(triton.cdiv(l, bl), i)](maps, grid, gk, gv, dmaps, dgrid, l, i, h, w, d,
                                            triton.next_power_of_2(d), bl, p, num_warps=8)
        dtags = (gk.float().reshape(l, i, p, d).sum((0, 2)).to(ctx.tags_dtype)
                 if ctx.has_tags and ctx.needs_input_grad[2] else None)
        return dmaps.to(maps.dtype), dgrid, dtags, None


def sample_tokens(maps, grid, tags=None, out_dtype=torch.bfloat16):
    """maps [I,h,w,2D] (keys then values, channels last), grid [I,L,P,2] normalized
    (x, y), P in {1, 2} points per image -> K, V [L, I*P, D]; token order (image, point);
    tags [I,D] added to K."""
    return _SampleTokens.apply(maps, grid, tags, out_dtype)


# ----------------------------------------------------------------- attention

@triton.jit
def _attend_fwd(Q, K, V, Bias, Null, Out, Lse, L, S, scale, D: tl.constexpr, DP: tl.constexpr,
                BL: tl.constexpr, BS: tl.constexpr):
    l = tl.program_id(0)*BL + tl.arange(0, BL)
    lm = l < L
    c = tl.arange(0, DP)
    cm = c < D
    q = tl.load(Q + l[:, None]*D + c[None, :], lm[:, None] & cm[None, :], 0.).to(tl.float32)*scale
    # The null token: zero key (score = its bias), zero value.
    m_i = tl.load(Null + l, lm, 0.).to(tl.float32)
    l_i = tl.full((BL,), 1., tl.float32)
    acc = tl.zeros((BL, DP), tl.float32)
    for s0 in range(0, S, BS):
        s = s0 + tl.arange(0, BS)
        sm = s < S
        off = (l[:, None, None]*S + s[None, :, None])*D + c[None, None, :]
        km = lm[:, None, None] & sm[None, :, None] & cm[None, None, :]
        k = tl.load(K + off, km, 0.).to(tl.float32)
        score = tl.sum(q[:, None, :]*k, 2)
        b = tl.load(Bias + l[:, None]*S + s[None, :], lm[:, None] & sm[None, :], float('-inf')).to(tl.float32)
        score = score + b
        m_new = tl.maximum(m_i, tl.max(score, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new[:, None])
        v = tl.load(V + off, km, 0.).to(tl.float32)
        acc = acc*alpha[:, None] + tl.sum(p[:, :, None]*v, 1)
        l_i = l_i*alpha + tl.sum(p, 1)
        m_i = m_new
    acc = acc/l_i[:, None]
    tl.store(Out + l[:, None]*D + c[None, :], acc.to(Out.dtype.element_ty), lm[:, None] & cm[None, :])
    tl.store(Lse + l, m_i + tl.log(l_i), lm)


@triton.jit
def _attend_bwd(Q, K, V, Bias, Null, Out, Lse, DO, DQ, DK, DV, DBias, DNull, L, S, scale,
                D: tl.constexpr, DP: tl.constexpr, BL: tl.constexpr, BS: tl.constexpr):
    l = tl.program_id(0)*BL + tl.arange(0, BL)
    lm = l < L
    c = tl.arange(0, DP)
    cm = c < D
    rm = lm[:, None] & cm[None, :]
    q = tl.load(Q + l[:, None]*D + c[None, :], rm, 0.).to(tl.float32)
    do = tl.load(DO + l[:, None]*D + c[None, :], rm, 0.).to(tl.float32)
    o = tl.load(Out + l[:, None]*D + c[None, :], rm, 0.).to(tl.float32)
    lse = tl.load(Lse + l, lm, 0.)
    delta = tl.sum(do*o, 1)
    p_null = tl.exp(tl.load(Null + l, lm, 0.).to(tl.float32) - lse)
    tl.store(DNull + l, -p_null*delta, lm)
    dq = tl.zeros((BL, DP), tl.float32)
    for s0 in range(0, S, BS):
        s = s0 + tl.arange(0, BS)
        sm = s < S
        off = (l[:, None, None]*S + s[None, :, None])*D + c[None, None, :]
        km = lm[:, None, None] & sm[None, :, None] & cm[None, None, :]
        k = tl.load(K + off, km, 0.).to(tl.float32)
        v = tl.load(V + off, km, 0.).to(tl.float32)
        bm = lm[:, None] & sm[None, :]
        b = tl.load(Bias + l[:, None]*S + s[None, :], bm, float('-inf')).to(tl.float32)
        score = tl.sum(q[:, None, :]*k, 2)*scale + b
        p = tl.exp(score - lse[:, None])
        dp = tl.sum(do[:, None, :]*v, 2)
        ds = p*(dp - delta[:, None])
        tl.store(DBias + l[:, None]*S + s[None, :], ds, bm)
        tl.store(DV + off, (p[:, :, None]*do[:, None, :]).to(DV.dtype.element_ty), km)
        tl.store(DK + off, (ds[:, :, None]*q[:, None, :]*scale).to(DK.dtype.element_ty), km)
        dq += tl.sum(ds[:, :, None]*k, 1)*scale
    tl.store(DQ + l[:, None]*D + c[None, :], dq.to(DQ.dtype.element_ty), rm)


# Launch shapes (rows per program, keys per inner step, warps); tuned on A100 for L=32768, S=96, D=48.
ATTEND_FWD = dict(BL=4, BS=32, num_warps=2)
ATTEND_BWD = dict(BL=4, BS=16, num_warps=2)


class _Attend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, bias, null):
        l, s, d = k.shape
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        bias = bias.contiguous().float()
        null = null.contiguous().float()
        out = torch.empty((l, d), device=q.device, dtype=q.dtype)
        lse = torch.empty((l,), device=q.device, dtype=torch.float32)
        scale = 1/math.sqrt(d)
        cfg = ATTEND_FWD
        _attend_fwd[(triton.cdiv(l, cfg['BL']),)](q, k, v, bias, null, out, lse, l, s, scale, d,
                                                 triton.next_power_of_2(d), cfg['BL'], cfg['BS'],
                                                 num_warps=cfg['num_warps'])
        ctx.save_for_backward(q, k, v, bias, null, out, lse)
        ctx.scale = scale
        ctx.dtypes = (bias.dtype, null.dtype)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, bias, null, out, lse = ctx.saved_tensors
        l, s, d = k.shape
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbias = torch.empty((l, s), device=q.device, dtype=torch.float32)
        dnull = torch.empty((l,), device=q.device, dtype=torch.float32)
        cfg = ATTEND_BWD
        _attend_bwd[(triton.cdiv(l, cfg['BL']),)](q, k, v, bias, null, out, lse, do.contiguous(), dq, dk, dv,
                                                 dbias, dnull, l, s, ctx.scale, d, triton.next_power_of_2(d),
                                                 cfg['BL'], cfg['BS'], num_warps=cfg['num_warps'])
        return dq, dk, dv, dbias, dnull


def attend(q, k, v, bias, null):
    """q [L,D], k/v [L,S,D], bias [L,S] (-inf = masked), null [L] -> [L,D].

    softmax over S keys plus a null token with zero key/value and score ``null``.
    """
    return _Attend.apply(q, k, v, bias, null)
