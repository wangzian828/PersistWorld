"""Coordinate algebra in FP32, outside BF16/FP16 autocast.

Casting operands with .float() alone is insufficient: autocast converts eligible matmul/einsum
operations back to BF16, and global camera translations amplify that rounding into large pixel
errors when two world transforms cancel.
"""
import torch

from . import host_sync as hs


def compose(left, right):
    """left @ right in FP32."""
    with torch.autocast(device_type=left.device.type, enabled=False):
        return left.float() @ right.float()


def solve(left, right):
    """inverse(left) @ right in FP32."""
    with torch.autocast(device_type=left.device.type, enabled=False):
        if left.is_cuda and hs.ENABLED:
            # Same factorization without reading the error flag back to the host.
            return torch.linalg.solve_ex(left.float(), right.float())[0]
        return torch.linalg.solve(left.float(), right.float())


def inverse(matrix):
    """Matrix inverse in FP32."""
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        if matrix.is_cuda and hs.ENABLED:
            return torch.linalg.inv_ex(matrix.float())[0]
        return torch.linalg.inv(matrix.float())


def transform_points(matrix, points):
    """[...,4,4] x [L,3] -> [...,L,3], preserving coordinate gradients."""
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        return (torch.einsum('...ij,lj->...li', matrix[..., :3, :3].float(), points.float())
                + matrix[..., :3, 3].float().unsqueeze(-2))


def project_homogeneous(matrix, points):
    """[...,4,4] x [L,4] -> [...,L,4]."""
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        return torch.einsum('...ij,lj->...li', matrix.float(), points.float())
