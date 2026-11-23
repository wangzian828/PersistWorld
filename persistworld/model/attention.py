"""FP32 multi-head attention with an explicit mathematical backward, used by the evolver.

Fused BF16 attention kernels showed large backward errors on saturated scene states, so the
evolver's attention runs a chunked FP32 softmax whose backward uses the normalized
probabilities. No key or value is omitted, rescaled or detached.
"""
import math

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

_SAFE_SOFTMAX = getattr(torch.ops.aten, '_safe_softmax', None)


def compatible_safe_softmax(scores):
    """Public-op equivalent of PyTorch's ``_safe_softmax`` (older versions): fully masked rows are zero."""
    result = torch.softmax(scores, dim=-1)
    return torch.where(torch.isneginf(scores).all(dim=-1, keepdim=True),
                       torch.zeros((), dtype=result.dtype, device=result.device), result)


def probabilities(query, scaled_key, factor, mask):
    """FP32 softmax probabilities of one query chunk; ``mask`` is boolean (True = keep) or additive."""
    scores = torch.matmul(query.float()*factor, scaled_key.transpose(-2, -1))
    if mask is not None:
        if mask.dtype == torch.bool:
            scores.masked_fill_(~mask, float('-inf'))
        else:
            scores.add_(mask.float())
    # Same safe softmax as PyTorch's mathematical SDPA, including masked rows.
    return (_SAFE_SOFTMAX.default(scores, -1) if _SAFE_SOFTMAX is not None
            else compatible_safe_softmax(scores))


def query_axis_product(left, right, chunk=0):
    """left.T @ right with independent FP32 partial sums along the long query axis."""
    length = left.shape[-2]
    if not chunk or length <= chunk:
        return torch.matmul(left.transpose(-2, -1), right)
    count, tail = divmod(length, chunk)
    used = count * chunk
    a = left[..., :used, :].reshape(*left.shape[:-2], count, chunk, left.shape[-1])
    b = right[..., :used, :].reshape(*right.shape[:-2], count, chunk, right.shape[-1])
    result = torch.matmul(a.transpose(-2, -1), b).sum(dim=-3, dtype=torch.float32)
    if tail:
        result.add_(torch.matmul(left[..., used:, :].transpose(-2, -1), right[..., used:, :]))
    return result


class AnalyticFP32Attention(torch.autograd.Function):
    """Chunked FP32 softmax attention; the backward recomputes the probabilities."""
    @staticmethod
    def forward(ctx, query, key, value, chunk, mask, reduction_chunk=0):
        ctx.save_for_backward(query, key, value)
        ctx.chunk, ctx.mask = chunk, mask
        ctx.reduction_chunk = reduction_chunk
        ctx.factor = math.sqrt(1.0/math.sqrt(query.shape[-1]))
        result = torch.empty_like(query)
        with torch.autocast(device_type=query.device.type, enabled=False):
            k = key.float()*ctx.factor
            v = value.float()
            for start in range(0, query.shape[-2], chunk):
                stop = start + chunk
                p = probabilities(query[..., start:stop, :], k, ctx.factor,
                                  None if mask is None else mask[..., start:stop, :])
                result[..., start:stop, :] = torch.matmul(p, v).to(query.dtype)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        query, key, value = ctx.saved_tensors
        dq = torch.empty_like(query)
        dk = torch.zeros_like(key, dtype=torch.float32)
        dv = torch.zeros_like(value, dtype=torch.float32)
        with torch.autocast(device_type=query.device.type, enabled=False):
            k = key.float()*ctx.factor
            v = value.float()
            for start in range(0, query.shape[-2], ctx.chunk):
                stop = start+ctx.chunk
                q = query[..., start:stop, :].float()
                g = upstream[..., start:stop, :].float()
                p = probabilities(q, k, ctx.factor, None if ctx.mask is None else ctx.mask[..., start:stop, :])
                dp = torch.matmul(g, v.transpose(-2, -1))
                ds = torch.ops.aten._softmax_backward_data.default(dp, p, -1, torch.float32)
                dq[..., start:stop, :] = (torch.matmul(ds, k)*ctx.factor).to(query.dtype)
                dk.add_(query_axis_product(ds, q*ctx.factor, ctx.reduction_chunk)*ctx.factor)
                dv.add_(query_axis_product(p, g, ctx.reduction_chunk))
        return (dq, dk.to(key.dtype), dv.to(value.dtype), None, None, None)[:len(ctx.needs_input_grad)]


class CachedAnalyticFP32Attention(torch.autograd.Function):
    """As AnalyticFP32Attention, keeping the FP32 probabilities for the backward instead of recomputing them."""
    @staticmethod
    def forward(ctx, query, key, value, chunk, mask):
        ctx.chunk = chunk
        ctx.factor = math.sqrt(1.0/math.sqrt(query.shape[-1]))
        cached = torch.empty((*query.shape[:-1], key.shape[-2]), device=query.device, dtype=torch.float32)
        result = torch.empty_like(query)
        with torch.autocast(device_type=query.device.type, enabled=False):
            k, v = key.float()*ctx.factor, value.float()
            for start in range(0, query.shape[-2], chunk):
                stop = start+chunk
                p = probabilities(query[..., start:stop, :], k, ctx.factor,
                                  None if mask is None else mask[..., start:stop, :])
                cached[..., start:stop, :] = p
                result[..., start:stop, :] = torch.matmul(p, v).to(query.dtype)
        ctx.save_for_backward(query, key, value, cached)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        query, key, value, cached = ctx.saved_tensors
        dq = torch.empty_like(query)
        dk = torch.zeros_like(key, dtype=torch.float32)
        dv = torch.zeros_like(value, dtype=torch.float32)
        with torch.autocast(device_type=query.device.type, enabled=False):
            k, v = key.float()*ctx.factor, value.float()
            for start in range(0, query.shape[-2], ctx.chunk):
                stop = start+ctx.chunk
                q = query[..., start:stop, :].float()
                g = upstream[..., start:stop, :].float()
                p = cached[..., start:stop, :]
                dp = torch.matmul(g, v.transpose(-2, -1))
                ds = torch.ops.aten._softmax_backward_data.default(dp, p, -1, torch.float32)
                dq[..., start:stop, :] = (torch.matmul(ds, k)*ctx.factor).to(query.dtype)
                dk.add_(torch.matmul(ds.transpose(-2, -1), q*ctx.factor)*ctx.factor)
                dv.add_(torch.matmul(p.transpose(-2, -1), g))
        return (dq, dk.to(key.dtype), dv.to(value.dtype), None, None)[:len(ctx.needs_input_grad)]


def bounded_math_multihead(module, query, memory, chunk=None, attn_mask=None):
    """nn.MultiheadAttention (batch_first) computed with FP32 attention and an explicit backward.

    A two-dimensional mask uses True for excluded keys, as nn.MultiheadAttention. ``chunk=None``
    bounds the score temporary by ``module.score_chunk_elements``. Execution attributes of the
    module: ``retain_fp32_probabilities`` (keep probabilities for the backward) and
    ``query_reduction_chunk`` (parallel dK/dV partial sums for short key sets).
    """
    if not (module.batch_first and module._qkv_same_embed_dim and module.dropout == 0
            and module.bias_k is None and module.bias_v is None and not module.add_zero_attn):
        raise ValueError('Unsupported attention configuration')
    if chunk is not None and (type(chunk) is not int or chunk < 1):
        raise ValueError('Positive query chunk required')
    width = module.embed_dim
    b, nq, _ = query.shape
    nk = memory.shape[1]
    heads = module.num_heads
    qweight, kvweight = module.in_proj_weight.split([width, 2*width])
    if module.in_proj_bias is None:
        qb = kvb = None
    else:
        qb, kvb = module.in_proj_bias.split([width, 2*width])
    if query is memory:
        # Self-attention: one fused input projection.
        qkv = F.linear(query.transpose(0, 1), module.in_proj_weight, module.in_proj_bias)
        qkv = qkv.unflatten(-1, (3, width)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv.unbind(0)
    else:
        q = F.linear(query.transpose(0, 1), qweight, qb)
        kv = F.linear(memory.transpose(0, 1), kvweight, kvb)
        kv = kv.unflatten(-1, (2, width)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        k, v = kv[0], kv[1]

    def arrange(x, length):
        return x.view(length, b*heads, width//heads).transpose(0, 1).view(b, heads, length, width//heads)
    q, k, v = arrange(q, nq), arrange(k, nk), arrange(v, nk)
    mask = None
    if attn_mask is not None:
        if attn_mask.requires_grad or attn_mask.shape not in ((nq, nk), (b, nq, nk)):
            raise ValueError('Expected a constant [Q,K] or [B,Q,K] mask')
        mask = ~attn_mask if attn_mask.dtype == torch.bool else attn_mask.float()
        if mask.ndim == 3:
            mask = mask[:, None]
    if chunk is None:
        elements = getattr(module, 'score_chunk_elements', 16*1024**2)
        chunk = max(1, min(nq, elements//max(1, math.prod(q.shape[:-2])*nk)))
    retain = (getattr(module, 'retain_fp32_probabilities', False) and torch.is_grad_enabled()
              and any(x.requires_grad for x in (q, k, v)))
    attention = CachedAnalyticFP32Attention if retain else AnalyticFP32Attention
    extra = ()
    reduction = getattr(module, 'query_reduction_chunk', 0)
    if reduction and not retain and nk <= 32 and nq >= 4096:
        extra = (reduction,)
    out = attention.apply(q, k, v, chunk, mask, *extra)
    out = out.permute(2, 0, 1, 3).contiguous().view(nq*b, width)
    out = F.linear(out, module.out_proj.weight, module.out_proj.bias)
    return out.view(nq, b, width).transpose(0, 1)
