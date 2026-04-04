import torch
from .. import util_torch

def make_vtx2morton_from_vtx2co(vtx2co: torch.Tensor, transform_co2unit: torch.Tensor):
    """Compute Morton codes for a set of 2D or 3D coordinates.

    Transforms each coordinate into the unit cube [0,1]^d and encodes it as a
    Morton (Z-order) code for use in BVH construction.

    Args:
        vtx2co: (num_vtx, 2 or 3) float32 - vertex coordinates
        transform_co2unit: (3x3 or 4x4) float32 - transformation mapping coordinates into the unit cube
    Returns:
        vtx2morton: (num_vtx,) uint32 - Morton code per vertex
    """
    num_vtx = vtx2co.shape[0]
    num_dim = vtx2co.shape[1]
    device = vtx2co.device
    #
    assert num_dim == 2 or num_dim == 3
    util_torch.assert_shape_dtype_device(vtx2co, shape=(num_vtx, num_dim), dtype=torch.float32, device=device)
    util_torch.assert_shape_dtype_device(transform_co2unit, shape=(num_dim+1, num_dim+1), dtype=torch.float32, device=device)
    #
    vtx2morton = torch.empty((num_vtx,), device=device, dtype=torch.uint32)
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Mortons

    Mortons.make_vtx2morton_from_vtx2co(
        util_torch.to_dlpack_safe(vtx2co, stream_ptr),
        util_torch.to_dlpack_safe(transform_co2unit.T.contiguous(), stream_ptr),
        util_torch.to_dlpack_safe(vtx2morton, stream_ptr),
        stream_ptr,
    )
    return vtx2morton


def make_bvhnodes_from_sorted_mortons(idx2obj: torch.Tensor, idx2morton: torch.Tensor):
    """Build a BVH tree from sorted Morton codes.

    Constructs a binary BVH with `2*n - 1` nodes from `n` objects sorted by
    their Morton codes. Each node stores (left_child, right_child, parent).

    Args:
        idx2obj: (n,) uint32 - object indices sorted by Morton code
        idx2morton: (n,) uint32 - corresponding sorted Morton codes
    Returns:
        bvhnodes: (2*n-1, 3) uint32 - BVH node data (left, right, parent) per node
    """
    n = idx2obj.shape[0]
    device = idx2obj.device
    #
    assert idx2obj.dtype == torch.uint32
    assert len(idx2obj.shape) == 1
    assert idx2morton.dtype == torch.uint32
    assert len(idx2morton.shape) == 1
    assert idx2obj.shape == idx2morton.shape
    #
    bvhnodes = torch.empty((n * 2 - 1, 3), device=device, dtype=torch.uint32)
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Mortons

    Mortons.make_bvhnodes_from_sorted_mortons(
        util_torch.to_dlpack_safe(idx2obj, stream_ptr),
        util_torch.to_dlpack_safe(idx2morton, stream_ptr),
        util_torch.to_dlpack_safe(bvhnodes, stream_ptr)
    )
    return bvhnodes
