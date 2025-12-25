import torch
from .. import util_torch

def vtx2morton_from_vtx2co(vtx2co: torch.Tensor, transform_co2unit: torch.Tensor):
    num_vtx = vtx2co.shape[0]
    num_dim = vtx2co.shape[1]
    device = vtx2co.device
    #
    assert len(vtx2co.shape) == 2
    assert num_dim == 2 or num_dim == 3
    assert vtx2co.dtype == torch.float32
    assert len(transform_co2unit.shape) == 2
    assert transform_co2unit.shape[0] == num_dim + 1
    assert transform_co2unit.shape[1] == num_dim + 1
    assert transform_co2unit.dtype == torch.float32
    #
    vtx2morton = torch.empty((num_vtx,), device=device, dtype=torch.uint32)
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Mortons

    Mortons.vtx2morton_from_vtx2co(
        util_torch.to_dlpack_safe(vtx2co, stream_ptr),
        util_torch.to_dlpack_safe(transform_co2unit.T.clone().contiguous(), stream_ptr),
        util_torch.to_dlpack_safe(vtx2morton, stream_ptr),
        stream_ptr,
    )
    return vtx2morton


def make_bvh(idx2obj: torch.Tensor, idx2morton: torch.Tensor):
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

    Mortons.make_bvh(
        util_torch.to_dlpack_safe(idx2obj, stream_ptr),
        util_torch.to_dlpack_safe(idx2morton, stream_ptr),
        util_torch.to_dlpack_safe(bvhnodes, stream_ptr)
    )
    return bvhnodes
