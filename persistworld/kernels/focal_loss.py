"""Triton forward and backward of the focal occupancy loss (``losses.focal_field_loss``).

One program handles 128 voxels and all classes. log(1 - p) of the winning class is computed from the
other classes (as ``losses.stable_log_probabilities``), so confidently wrong logits stay finite.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ..losses import validate_free_weight
from ..model import host_sync as hs


# Element offset of flat voxel ``rows`` in a [B,X,Y,Z] layout with strides S0..S3 (FAST_X: x varies fastest).
@triton.jit
def _offset(rows, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr, FAST_X: tl.constexpr):
    if FAST_X:
        return (rows//(X*Y*Z))*S0+(rows%X)*S1+(rows//X%Y)*S2+(rows//(X*Y)%Z)*S3
    return (rows//(X*Y*Z))*S0+(rows//(Y*Z)%X)*S1+(rows//Z%Y)*S2+(rows%Z)*S3


# Sum of the per-block partial losses and counts -> TOTAL (valid voxels) and LOSS (their mean).
@triton.jit
def _finish(P, D, TOTAL, LOSS, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    n = tl.sum(tl.load(D+i, i < N, other=0), 0).to(tl.float32)
    v = tl.sum(tl.load(P+i, i < N, other=0.), 0)
    tl.store(TOTAL, n)
    tl.store(LOSS, v/tl.maximum(n, 1.))


# Forward (BACK=False): per-block loss sums P and valid counts D. Backward (BACK=True): logit gradients G,
# scaled by the upstream gradient UP and the valid count TOTAL of the forward.
@triton.jit
def _kernel(L, A, M, WP, WN, P, D, TOTAL, UP, G, N: tl.constexpr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr, S4: tl.constexpr,
            G0: tl.constexpr, G1: tl.constexpr, G2: tl.constexpr, G3: tl.constexpr, G4: tl.constexpr,
            SCALE: tl.constexpr, FREE_WEIGHT: tl.constexpr, FREE_ID: tl.constexpr,
            BACK: tl.constexpr, C: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
    block = tl.program_id(0)
    r = block*R+tl.arange(0, R)
    c = tl.arange(0, K)
    idx = _offset(r, X, Y, Z, X*Y*Z, Y*Z, Z, 1, S1 < S3)
    y = tl.load(A+idx, r < N, other=-1)
    valid = (r < N) & tl.load(M+idx, r < N, other=0) & (y >= 0) & (y < C)
    base = _offset(r, X, Y, Z, S0, S1, S2, S3, S1 < S3)
    x = tl.load(L+base[:, None]+c[None, :]*S4, (r[:, None] < N) & (c[None, :] < C), other=0).to(tl.float32)
    x = tl.where(c[None, :] < C, x, -float('inf'))
    winner = tl.argmax(x, 1)
    top = c[None, :] == winner[:, None]
    maximum = tl.max(x, 1)
    shift = x-maximum[:, None]
    ex = tl.exp(shift)
    den = tl.sum(ex, 1)
    ld = tl.log(den)
    logp = shift-ld[:, None]
    p = tl.exp(logp)
    q = -libdevice.expm1(logp)
    second = tl.max(tl.where(top, -float('inf'), x), 1)
    other_shift = tl.where(top, -float('inf'), x-second[:, None])
    other_ex = tl.exp(other_shift)
    other_den = tl.sum(other_ex, 1)
    winning_logq = second-maximum+tl.log(other_den)-ld
    ordinary_logq = tl.log(tl.maximum(den[:, None]-ex, 0.5))-ld[:, None]
    logq = tl.where(top, winning_logq[:, None], ordinary_logq)
    is_y = c[None, :] == y[:, None]
    logpt = tl.sum(tl.where(is_y, logp, 0.), 1)
    apt = tl.load(WP+y, valid, other=0.).to(tl.float32)
    an = tl.load(WN+c, c < C, other=0.).to(tl.float32)
    pt = tl.exp(logpt)
    qt = -libdevice.expm1(logpt)
    if BACK:
        positive_factor = apt*qt*(qt-2*pt*logpt)
        positive = tl.where(is_y, -qt[:, None]*positive_factor[:, None], p*positive_factor[:, None])
        diag = tl.where(is_y, 0., an[None, :]*p*p*(p-2*q*logq))
        # The winning class may have q rounded to zero. Its off-diagonal
        # derivative uses the conditional softmax over all other classes.
        ordinary_ratio = tl.where(top, 0., diag/tl.maximum(q, 0.5))
        total_ratio = tl.sum(ordinary_ratio, 1)
        top_diag = tl.sum(tl.where(top, diag, 0.), 1)
        conditional = other_ex/other_den[:, None]
        negative = diag+p*ordinary_ratio-p*total_ratio[:, None]-top_diag[:, None]*conditional
        scale = tl.load(UP)*SCALE/tl.maximum(tl.load(TOTAL), 1.)
        grad = tl.where(valid[:, None], (positive+negative)*scale, 0.)
        if FREE_WEIGHT != 1.:
            grad = grad*tl.where(y == FREE_ID, FREE_WEIGHT, 1.)[:, None]
        dest = _offset(r, X, Y, Z, G0, G1, G2, G3, S1 < S3)
        tl.store(G+dest[:, None]+c[None, :]*G4, grad, (r[:, None] < N) & (c[None, :] < C))
    else:
        negative = tl.sum(tl.where(is_y, 0., -an[None, :]*p*p*logq), 1)
        value = (-apt*qt*qt*logpt+negative)*SCALE
        if FREE_WEIGHT != 1.:
            value = value*tl.where(y == FREE_ID, FREE_WEIGHT, 1.)
        tl.store(P+block, tl.sum(tl.where(valid, value, 0.), 0))
        tl.store(D+block, tl.sum(valid.to(tl.int32), 0))


class _FocalField(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, mask, wp, wn, scale, free_weight, free_id):
        labels = labels.contiguous()
        mask = mask.bool().contiguous()
        n = labels.numel()
        _, x, y, z, c = logits.shape
        blocks = triton.cdiv(n, 128)
        parts = torch.empty(blocks, device=logits.device, dtype=torch.float32)
        counts = torch.empty(blocks, device=logits.device, dtype=torch.int32)
        total = torch.empty((), device=logits.device, dtype=torch.float32)
        out = torch.empty_like(total)
        _kernel[(blocks,)](logits, labels, mask, wp, wn, parts, counts, total, total, logits, n, x, y, z,
                           *logits.stride(), *logits.stride(), scale, free_weight, free_id, False, c, 128,
                           triton.next_power_of_2(c), num_warps=4)
        _finish[(1,)](parts, counts, total, out, blocks, triton.next_power_of_2(blocks), num_warps=4)
        ctx.save_for_backward(logits, labels, mask, wp, wn, total)
        ctx.scale = scale
        ctx.free_weight = free_weight
        ctx.free_id = free_id
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        logits, labels, mask, wp, wn, total = ctx.saved_tensors
        result = torch.empty_like(logits)
        _, x, y, z, c = logits.shape
        n = labels.numel()
        _kernel[(triton.cdiv(n, 128),)](logits, labels, mask, wp, wn, total, total, total, grad, result, n, x, y, z,
                                        *logits.stride(), *result.stride(), ctx.scale, ctx.free_weight, ctx.free_id,
                                        True, c, 128, triton.next_power_of_2(c), num_warps=4)
        return result, None, None, None, None, None, None, None


def fused_focal_field_loss(logits, labels, mask, positive_weights, negative_weights, scale, free_target_weight,
                           free_id):
    """Same arguments and value as ``losses.focal_field_loss`` (CUDA tensors only)."""
    if (logits.ndim != 5 or not 2 <= logits.shape[-1] <= 256 or logits.shape[:-1] != labels.shape
            or labels.shape != mask.shape):
        raise ValueError('Expected [B,X,Y,Z,C] with matching targets')
    if not logits.is_cuda or logits.device != labels.device or logits.device != mask.device:
        raise ValueError('Fused focal field requires one CUDA device')
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Unsupported logit dtype')
    validate_free_weight(free_target_weight, free_id, logits.shape[-1])
    if labels.dtype not in (torch.int32, torch.int64, torch.uint8):
        raise ValueError('Integer labels required')
    if not labels.numel():
        return logits.float().sum()*0
    if isinstance(positive_weights, torch.Tensor) or not hs.ENABLED:
        wp = torch.as_tensor(positive_weights, device=logits.device, dtype=torch.float32).contiguous()
        wn = torch.as_tensor(negative_weights, device=logits.device, dtype=torch.float32).contiguous()
    else:
        wp = hs.const([float(x) for x in positive_weights], device=logits.device)
        wn = hs.const([float(x) for x in negative_weights], device=logits.device)
    if wp.ndim != 1 or wn.shape != wp.shape or wp.numel() != logits.shape[-1]:
        raise ValueError('Weight shape differs')
    return _FocalField.apply(logits, labels, mask, wp, wn, float(scale), float(free_target_weight), free_id)
