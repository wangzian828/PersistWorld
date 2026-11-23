"""Fused LN((1 + scale) * LN(x) + shift) without affine weights (the evolver block's modulated norm).

Statistics and reductions are FP32; the 1+scale rounding follows the input dtype. CUDA with FP32
state, or BF16 state under autocast, uses the Triton kernels of ``fused_norm_kernels``; every
other case (including CPU) uses the PyTorch reference.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def reference(x, scale, shift, eps1=1e-5, eps2=1e-5):
    """PyTorch reference: LN((1 + scale) * LN(x) + shift)."""
    return F.layer_norm(F.layer_norm(x, (x.shape[-1],), eps=eps1)*(1+scale)+shift,
                        (x.shape[-1],), eps=eps2)


class _FusedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift, eps1, eps2):
        from .fused_norm_kernels import launch_forward
        out, stats = launch_forward(x, scale, shift, eps1, eps2)
        ctx.save_for_backward(x, scale, shift, stats)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        from .fused_norm_kernels import launch_backward
        x, s, t, stats = ctx.saved_tensors
        return *launch_backward(x, s, t, stats, upstream), None, None


def normalized_modulation(x, scale, shift, eps1=1e-5, eps2=1e-5):
    """LN((1 + scale) * LN(x) + shift) over the last dimension; the Triton kernels where they apply."""
    if x.shape != scale.shape or x.shape != shift.shape or x.device != scale.device or x.device != shift.device:
        raise ValueError('Norm fusion requires identical shapes and devices')
    if (x.dtype not in (torch.float32, torch.bfloat16) or scale.dtype not in (torch.float32, torch.bfloat16)
            or shift.dtype != scale.dtype):
        raise ValueError('Norm fusion requires FP32/BF16 state and matching FP32/BF16 scale/shift')
    if not x.ndim or not 1 <= x.shape[-1] <= 4096 or x.numel() == 0:
        raise ValueError('Unsupported normalization dimensions')
    if not x.is_cuda or (x.dtype == torch.bfloat16 and not torch.is_autocast_enabled('cuda')):
        return reference(x, scale, shift, eps1, eps2)
    return _FusedNorm.apply(x, scale, shift, eps1, eps2)
