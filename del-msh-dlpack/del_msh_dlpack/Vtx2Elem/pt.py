import torch
from .. import _CapsuleAsDLPack


def from_uniform_mesh(elem2vtx: torch.Tensor, num_vtx: int):
    """make vertex surrounding vertex data from uniform mesh"""
    device = elem2vtx.device
    assert len(elem2vtx.shape) == 2
    assert elem2vtx.dtype == torch.uint32
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Vtx2Elem

    cap_vtx2idx, cap_idx2elem = Vtx2Elem.from_uniform_mesh(
        elem2vtx.__dlpack__(), num_vtx, stream_ptr
    )
    vtx2idx = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2idx)).clone()
    idx2elem = torch.from_dlpack(_CapsuleAsDLPack(cap_idx2elem)).clone()
    return vtx2idx, idx2elem
