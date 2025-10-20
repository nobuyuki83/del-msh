import torch
from .. import util_torch

def permute(old2val: torch.Tensor, new2old: torch.Tensor):
    n = old2val.shape[0]
    device = old2val.device
    #
    assert old2val.dtype == torch.uint32
    assert len(old2val.shape) == 1
    assert new2old.dtype == torch.uint32
    assert new2old.device == device
    assert new2old.shape == old2val.shape
    #
    new2val = torch.empty(size=(n,), device=device, dtype=torch.uint32)
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Array1D

    Array1D.permute(
        util_torch.to_dlpack_safe(old2val),
        util_torch.to_dlpack_safe(new2old),
        util_torch.to_dlpack_safe(new2val),
        stream_ptr,
    )
    return new2val


def argsort(idx2val: torch.Tensor):
    n = idx2val.shape[0]
    device = idx2val.device
    #
    jdx2idx = torch.empty(size=(n,), device=device, dtype=torch.uint32)
    jdx2val = idx2val.clone()
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Array1D

    Array1D.argsort(
        util_torch.to_dlpack_safe(jdx2val),
        util_torch.to_dlpack_safe(jdx2idx),
        stream_ptr,
    )
    return jdx2idx, jdx2val
