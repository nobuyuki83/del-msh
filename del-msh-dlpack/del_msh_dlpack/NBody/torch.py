import torch
from .. import util_torch

def screened_poisson(
        vtx2co: torch.Tensor,
        vtx2rhs: torch.Tensor,
        lambda_param: float,
        epsilon: float,
        wtx2co: torch.Tensor):
    num_vtx = vtx2co.shape[0]
    num_wtx = wtx2co.shape[0]
    device = vtx2co.device
    #
    assert vtx2co.shape == (num_vtx, 3) and vtx2rhs.dtype == torch.float32
    assert vtx2rhs.shape == (num_vtx, 3) and vtx2rhs.device == device and vtx2rhs.dtype == torch.float32
    assert wtx2co.shape == (num_wtx, 3) and wtx2co.device == device and wtx2co.dtype == torch.float32
    #
    wtx2lhs = torch.zeros(size=(num_wtx, 3), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import NBody

    NBody.screened_poisson(
        util_torch.to_dlpack_safe(vtx2co, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        util_torch.to_dlpack_safe(wtx2co, stream_ptr),
        util_torch.to_dlpack_safe(wtx2lhs, stream_ptr),
        lambda_param,
        epsilon,
        stream_ptr
    )

    return wtx2lhs


def elastic(
        vtx2co: torch.Tensor,
        vtx2rhs: torch.Tensor,
        nu: float,
        epsilon: float,
        wtx2co: torch.Tensor):
    num_vtx = vtx2co.shape[0]
    num_wtx = wtx2co.shape[0]
    device = vtx2co.device
    #
    assert vtx2co.shape == (num_vtx, 3) and vtx2rhs.dtype == torch.float32
    assert vtx2rhs.shape == (num_vtx, 3) and vtx2rhs.device == device and vtx2rhs.dtype == torch.float32
    assert wtx2co.shape == (num_wtx, 3) and wtx2co.device == device and wtx2co.dtype == torch.float32
    #
    wtx2lhs = torch.zeros(size=(num_wtx, 3), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import NBody

    NBody.elastic(
        util_torch.to_dlpack_safe(vtx2co, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        util_torch.to_dlpack_safe(wtx2co, stream_ptr),
        util_torch.to_dlpack_safe(wtx2lhs, stream_ptr),
        nu,
        epsilon,
        stream_ptr
    )

    return wtx2lhs