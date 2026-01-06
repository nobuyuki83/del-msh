import torch
from .. import util_torch

def elem2volume(
    elem2idx_offset: torch.Tensor,
    idx2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor):
    #
    num_elem = elem2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    device = elem2idx_offset.device
    #
    util_torch.assert_shape_dtype_device(elem2idx_offset, (num_elem+1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx,3), torch.float32, device)
    #
    elem2volume = torch.empty(num_elem, dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import PolyhedronMesh

    PolyhedronMesh.elem2volume(
        util_torch.to_dlpack_safe(elem2idx_offset, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(elem2volume, stream_ptr),
        stream_ptr
    )
    return elem2volume








