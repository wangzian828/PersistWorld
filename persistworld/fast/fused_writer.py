"""Inference-only fused writer chunk: sampling, ray-depth geometry embeddings and attention in Triton.

The training path (kernels/fused_evidence.py) materializes per-(cell, token) keys/values and adds the
geometry MLP outputs in separate operations. Here the keys/values are built in one kernel (the geometry
MLPs with BF16 weights and BF16-rounded inputs, hidden and outputs, FP32 accumulation: the autocast
roundings of the reference), skipping blocks whose tokens are all masked, and the attention kernel
loads only unmasked tokens. Same parameters, tokens and masks; only the summation order differs.
"""
import math
import types

import torch
import triton
import triton.language as tl

from ..kernels.fused_evidence import _corner_setup
from ..model import ray_depth
from ..model.geometry import project_homogeneous


@triton.jit
def _gelu_bf16(x):
    x = x.to(tl.bfloat16).to(tl.float32)
    y = 0.5*x*(1.0 + tl.math.erf(x*0.7071067811865476))
    return y.to(tl.bfloat16)


@triton.jit
def _tokens_skip_fwd(Maps, Grid, Tags, Geo, Bias, W1k, B1k, W2k, B2k, W1v, B1v, W2v, B2v, K, V,
                     L, H, W, D: tl.constexpr, DP: tl.constexpr, S: tl.constexpr, BM: tl.constexpr,
                     LOWP: tl.constexpr, P: tl.constexpr):
    """Token-major keys/values as fused_evidence.sample_tokens plus the geometry embeddings, one token s
    per program column and BM cells per program. Blocks whose BM cells all mask token s (bias -inf) are
    skipped: nothing reads them (attend_skip loads keys/values only where the bias is finite)."""
    l = tl.program_id(0)*BM + tl.arange(0, BM)
    s = tl.program_id(1)
    lm = l < L
    b = tl.load(Bias + l*S + s, lm, float('-inf'))
    if tl.max(b, 0) > float('-inf'):
        i = s // P
        pair = s % P
        c = tl.arange(0, DP)
        cm = c < D
        fx, fy, x0, y0 = _corner_setup(Grid, i, l, lm, pair, L, H, W, P)
        acc_k = tl.zeros((BM, DP), tl.float32)
        acc_v = tl.zeros((BM, DP), tl.float32)
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
        acc_k += tl.load(Tags + i*D + c, cm, 0.).to(tl.float32)[None, :]
        if LOWP:
            acc_k = acc_k.to(tl.bfloat16).to(tl.float32)
            acc_v = acc_v.to(tl.bfloat16).to(tl.float32)
        r = l*S + s
        g = Geo + r*4
        g0 = tl.load(g + 0, lm, 0.).to(tl.bfloat16).to(tl.float32)[:, None]
        g1 = tl.load(g + 1, lm, 0.).to(tl.bfloat16).to(tl.float32)[:, None]
        g2 = tl.load(g + 2, lm, 0.).to(tl.bfloat16).to(tl.float32)[:, None]
        g3 = tl.load(g + 3, lm, 0.).to(tl.bfloat16).to(tl.float32)[:, None]
        kk = tl.arange(0, DP)
        wm = (kk[:, None] < D) & (c[None, :] < D)
        om = lm[:, None] & cm[None, :]
        out = r[:, None]*D + c[None, :]
        h = (g0*tl.load(W1k + c*4 + 0, cm, 0.).to(tl.float32)[None, :]
             + g1*tl.load(W1k + c*4 + 1, cm, 0.).to(tl.float32)[None, :]
             + g2*tl.load(W1k + c*4 + 2, cm, 0.).to(tl.float32)[None, :]
             + g3*tl.load(W1k + c*4 + 3, cm, 0.).to(tl.float32)[None, :]
             + tl.load(B1k + c, cm, 0.).to(tl.float32)[None, :])
        e = (tl.dot(_gelu_bf16(h), tl.load(W2k + c[None, :]*D + kk[:, None], wm, 0.))
             + tl.load(B2k + c, cm, 0.).to(tl.float32)[None, :])
        key = acc_k + e.to(tl.bfloat16).to(tl.float32)
        tl.store(K + out, key.to(K.dtype.element_ty), om)
        h = (g0*tl.load(W1v + c*4 + 0, cm, 0.).to(tl.float32)[None, :]
             + g1*tl.load(W1v + c*4 + 1, cm, 0.).to(tl.float32)[None, :]
             + g2*tl.load(W1v + c*4 + 2, cm, 0.).to(tl.float32)[None, :]
             + g3*tl.load(W1v + c*4 + 3, cm, 0.).to(tl.float32)[None, :]
             + tl.load(B1v + c, cm, 0.).to(tl.float32)[None, :])
        e = (tl.dot(_gelu_bf16(h), tl.load(W2v + c[None, :]*D + kk[:, None], wm, 0.))
             + tl.load(B2v + c, cm, 0.).to(tl.float32)[None, :])
        value = acc_v + e.to(tl.bfloat16).to(tl.float32)
        tl.store(V + out, value.to(V.dtype.element_ty), om)


@triton.jit
def _attend_skip_fwd(Q, K, V, Bias, Null, Out, L, scale, D: tl.constexpr, DP: tl.constexpr,
                     S: tl.constexpr, BL: tl.constexpr, BS: tl.constexpr):
    """fused_evidence._attend_fwd, loading keys/values only for tokens with a finite bias."""
    l = tl.program_id(0)*BL + tl.arange(0, BL)
    lm = l < L
    c = tl.arange(0, DP)
    cm = c < D
    q = tl.load(Q + l[:, None]*D + c[None, :], lm[:, None] & cm[None, :], 0.).to(tl.float32)*scale
    m_i = tl.load(Null + l, lm, 0.).to(tl.float32)
    l_i = tl.full((BL,), 1., tl.float32)
    acc = tl.zeros((BL, DP), tl.float32)
    for s0 in range(0, S, BS):
        s = s0 + tl.arange(0, BS)
        sm = s < S
        b = tl.load(Bias + l[:, None]*S + s[None, :], lm[:, None] & sm[None, :], float('-inf')).to(tl.float32)
        if tl.max(tl.max(b, 1), 0) > float('-inf'):
            ok = b > float('-inf')
            off = (l[:, None, None]*S + s[None, :, None])*D + c[None, None, :]
            km = ok[:, :, None] & cm[None, None, :]
            k = tl.load(K + off, km, 0.).to(tl.float32)
            score = tl.sum(q[:, None, :]*k, 2) + b
            m_new = tl.maximum(m_i, tl.max(score, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(score - m_new[:, None])
            v = tl.load(V + off, km, 0.).to(tl.float32)
            acc = acc*alpha[:, None] + tl.sum(p[:, :, None]*v, 1)
            l_i = l_i*alpha + tl.sum(p, 1)
            m_i = m_new
    acc = acc/l_i[:, None]
    tl.store(Out + l[:, None]*D + c[None, :], acc.to(Out.dtype.element_ty), lm[:, None] & cm[None, :])


# Launch shapes: cells per program and warps of the token kernel; rows, keys per step and warps of attention.
SKIP = dict(BM=64, tok_warps=4, BL=16, BS=2, att_warps=4)


def _weights(mlp):
    """BF16 weights and biases of a two-layer geometry MLP (the casts autocast applies in the reference)."""
    a, b = mlp[0], mlp[2]
    bf = torch.bfloat16
    return (a.weight.to(bf).contiguous(), a.bias.to(bf).contiguous(), b.weight.to(bf).contiguous(),
            b.bias.to(bf).contiguous())


# Opaque custom ops so torch.compile can fuse the per-cell work around the two kernels.
@torch.library.custom_op('persistworld::tokens_skip', mutates_args=())
def tokens_skip_op(maps: torch.Tensor, grid: torch.Tensor, tags: torch.Tensor, geo: torch.Tensor,
                   bias: torch.Tensor, w1k: torch.Tensor, b1k: torch.Tensor, w2k: torch.Tensor, b2k: torch.Tensor,
                   w1v: torch.Tensor, b1v: torch.Tensor, w2v: torch.Tensor, b2v: torch.Tensor,
                   lowp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    i, h, w, d2 = maps.shape
    d = d2//2
    l = grid.shape[1]
    p = grid.shape[2]
    s = i*p
    k = torch.empty((l, s, d), device=maps.device, dtype=torch.bfloat16 if lowp else torch.float32)
    v = torch.empty_like(k)
    bm = SKIP['BM']
    _tokens_skip_fwd[(triton.cdiv(l, bm), s)](maps.contiguous(), grid.contiguous().float(), tags.contiguous(),
                                              geo.contiguous().float(), bias.contiguous().float(),
                                              w1k, b1k, w2k, b2k, w1v, b1v, w2v, b2v, k, v, l, h, w, d,
                                              triton.next_power_of_2(d), s, bm, lowp, p,
                                              num_warps=SKIP['tok_warps'])
    return k, v


@tokens_skip_op.register_fake
def _(maps, grid, tags, geo, bias, w1k, b1k, w2k, b2k, w1v, b1v, w2v, b2v, lowp):
    i, h, w, d2 = maps.shape
    l = grid.shape[1]
    k = maps.new_empty((l, i*grid.shape[2], d2//2), dtype=torch.bfloat16 if lowp else torch.float32)
    return k, torch.empty_like(k)


@torch.library.custom_op('persistworld::attend_skip', mutates_args=())
def attend_skip_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                   null: torch.Tensor) -> torch.Tensor:
    l, s, d = k.shape
    out = torch.empty((l, d), device=q.device, dtype=q.dtype)
    _attend_skip_fwd[(triton.cdiv(l, SKIP['BL']),)](q.contiguous(), k, v, bias.contiguous().float(),
                                                    null.contiguous().float(), out, l, 1/math.sqrt(d), d,
                                                    triton.next_power_of_2(d), s, SKIP['BL'], SKIP['BS'],
                                                    num_warps=SKIP['att_warps'])
    return out


@attend_skip_op.register_fake
def _(q, k, v, bias, null):
    return q.new_empty((k.shape[0], k.shape[2]))


def fast_chunk(self, prior, xyz, key_maps, image_from_scene, image_hw, fresh, tags, patch_grid, depth_volume=None):
    """CameraWriter._chunk for inference on CUDA (fused evidence path, no ablation)."""
    assert not torch.is_grad_enabled() and self._fused(prior) and not self.ablate_attention and not self.ablate_depth
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
        grid = torch.stack([(uv[..., 0]+.5)/image_hw[1], (uv[..., 1]+.5)/image_hw[0]], -1)*2-1
    offset = self.offset(prior[:, None].expand(-1, t, -1, -1)).tanh()
    o = offset[:, :, None, :, None]
    grid = grid+torch.stack([o[..., 0]*(2/w), o[..., 1]*(2/h)], -1)
    valid = (depth > 1e-5) & (grid.abs() < 1).all(-1) & fresh[:, :, None, None, None]
    valid = valid.permute(0, 3, 1, 2, 4)
    bias = (torch.zeros(valid.shape, device=valid.device, dtype=torch.float32)
            .masked_fill(~valid, -torch.inf).reshape(b, count, -1))
    depth_bias, geometry = ray_depth.sample_geometry(self, grid, depth, depth_volume, b, t, cameras, count)
    bias = bias+depth_bias.to(bias.dtype)
    null = torch.zeros_like(bias[:, :, :1])
    grid_s = grid.reshape(b*t*cameras, count, 1, 2).float()
    tags_s = tags.reshape(t*cameras, d)
    bias0 = bias[0].contiguous()
    keys, values = torch.ops.persistworld.tokens_skip(key_maps, grid_s, tags_s, geometry[0], bias0, *self._geo_weights,
                                                      prior.dtype == torch.bfloat16)
    candidate = prior
    for norm, qproj, output, ff in zip(self.norm, self.query, self.output, self.ff):
        query = qproj(norm(candidate)).reshape(count, d)
        evidence = torch.ops.persistworld.attend_skip(query, keys, values, bias0, null[0, :, 0]).reshape(b, count, d)
        candidate = candidate+output(evidence)
        candidate = candidate+ff(candidate)
    seen = valid.flatten(2).any(-1)
    gate = self.gate(torch.cat([prior, candidate], -1))*seen[..., None]
    return prior+gate*(candidate-prior), seen


def install(model):
    """Replace the writer chunk of an eval camera model by ``fast_chunk``."""
    writer = model.writer
    writer._chunk = types.MethodType(fast_chunk, writer)
    # The BF16 casts autocast would apply at every call, made once.
    writer._geo_weights = (*_weights(writer.geometry_key), *_weights(writer.geometry_value))
