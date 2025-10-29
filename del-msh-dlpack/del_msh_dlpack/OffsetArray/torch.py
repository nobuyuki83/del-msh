import torch
from .. import util_torch

def aggregate(
    idx2jdx_offset: torch.Tensor,
    jdx2kdx: torch.Tensor,
    kdx2val: torch.Tensor):
    num_idx = idx2jdx_offset.shape[0] - 1
    num_jdx = jdx2kdx.shape[0]
    device = idx2jdx_offset.device
    num_dim = kdx2val.shape[1]
    #
    assert idx2jdx_offset.shape == (num_idx+1,)
    assert jdx2kdx.shape == (num_jdx,) and jdx2kdx.device == device and jdx2kdx.dtype == torch.uint32
    assert kdx2val.shape == (num_jdx,num_dim) and kdx2val.device == device and kdx2val.dtype == torch.float32
    #
    idx2aggval = torch.zeros(size=(num_idx,num_dim), device=device, dtype=torch.float32)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import OffsetArray

    OffsetArray.aggregate(
        util_torch.to_dlpack_safe(idx2jdx_offset, stream_ptr),
        util_torch.to_dlpack_safe(jdx2kdx, stream_ptr),
        util_torch.to_dlpack_safe(kdx2val, stream_ptr),
        util_torch.to_dlpack_safe(idx2aggval, stream_ptr),
        stream_ptr)

    return idx2aggval