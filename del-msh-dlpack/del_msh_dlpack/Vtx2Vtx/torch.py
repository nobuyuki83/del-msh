import torch
from .. import _CapsuleAsDLPack
from .. import util_torch


def from_uniform_mesh(elem2vtx: torch.Tensor, num_vtx: int, is_self: bool):
    """make vertex surrounding vertex data from uniform mesh"""
    device = elem2vtx.device
    assert len(elem2vtx.shape) == 2
    assert elem2vtx.dtype == torch.uint32
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Vtx2Vtx

    cap_vtx2idx, cap_idx2vtx = Vtx2Vtx.from_uniform_mesh(
        util_torch.to_dlpack_safe(elem2vtx, stream_ptr),
        num_vtx, is_self, stream_ptr
    )
    vtx2idx = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2idx))
    idx2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_idx2vtx))
    return vtx2idx, idx2vtx


def laplacian_smoothing(
    vtx2idx: torch.Tensor,
    idx2vtx: torch.Tensor,
    lambda0: float,
    vtx2lhs: torch.Tensor,
    vtx2rhs: torch.Tensor,
    num_iter: int,
    vtx2lhstmp: torch.Tensor | None,
):
    """Solve the linear system from screened Poisson equation using Jacobi method:
    [I + lambda * L] {vtx2lhs} = {vtx2rhs}
    where L = [ .., -1, .., valence, ..,-1, .. ]
    """
    device = vtx2idx.device
    num_vtx = vtx2idx.shape[0] - 1
    if vtx2lhstmp is None:
        vtx2lhstmp = torch.zeros_like(vtx2lhs)
    #
    assert len(vtx2idx.shape) == 1
    assert len(idx2vtx.shape) == 1
    assert vtx2lhs.shape == vtx2rhs.shape
    assert vtx2idx.dtype == torch.uint32
    assert idx2vtx.dtype == torch.uint32
    assert len(vtx2lhs.shape) == 2
    assert vtx2lhs.shape[0] == num_vtx
    assert vtx2lhs.shape == vtx2rhs.shape
    assert vtx2lhs.dtype == vtx2rhs.dtype == torch.float32
    assert num_iter >= 0
    assert idx2vtx.device == device, "idx2vtx should be on the same device as vtx2idx"
    assert vtx2rhs.device == device, "vtx2rhs should be on the same device as vtx2idx"
    assert vtx2lhs.device == device, "vtx2lhs should be on the same device as vtx2lhs"
    assert vtx2lhstmp.device == device, (
        "the vtx2lhstmp should be on the same device as vtx2idx"
    )
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import Vtx2Vtx

    Vtx2Vtx.laplacian_smoothing(
        util_torch.to_dlpack_safe(vtx2idx, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        lambda0,
        util_torch.to_dlpack_safe(vtx2lhs, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        num_iter,
        util_torch.to_dlpack_safe(vtx2lhstmp, stream_ptr),
        stream_ptr,
    )


def multiply_graph_laplacian(
    vtx2idx: torch.Tensor, idx2vtx: torch.Tensor, vtx2rhs: torch.Tensor
) -> torch.Tensor:
    assert len(vtx2idx.shape) == 1
    assert vtx2idx.dtype == torch.uint32
    assert len(idx2vtx.shape) == 1
    assert idx2vtx.dtype == torch.uint32
    assert len(vtx2rhs.shape) == 2
    assert vtx2rhs.dtype == torch.float32
    #
    vtx2lhs = torch.zeros_like(vtx2rhs)
    #
    from .. import Vtx2Vtx

    Vtx2Vtx.multiply_graph_laplacian(
        vtx2idx.__dlpack__(),
        idx2vtx.__dlpack__(),
        vtx2rhs.__dlpack__(),
        vtx2lhs.__dlpack__(),
    )
    return vtx2lhs
