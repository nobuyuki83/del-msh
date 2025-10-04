from typing import Any

import torch

def tri2normal(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor):
    num_tri = tri2vtx.shape[0]
    device = tri2vtx.device
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    assert tri2vtx.dtype == torch.uint32
    assert tri2vtx.is_contiguous()
    assert tri2vtx.requires_grad == False
    #
    assert len(vtx2xyz.shape) == 2
    assert vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == torch.float32
    assert vtx2xyz.is_contiguous()
    assert vtx2xyz.device == device, "vtx2xyz should be on the same device as tri2vtx"
    assert vtx2xyz.requires_grad == False
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
        #print(device, stream_ptr)
    #
    tri2nrm = torch.empty(size=(num_tri,3), dtype=torch.float32, device=device)
    from .. import TriMesh3
    TriMesh3.tri2normal( 
        tri2vtx.__dlpack__(stream=stream_ptr),
        vtx2xyz.__dlpack__(stream=stream_ptr),
        tri2nrm.__dlpack__(stream=stream_ptr),
        stream_ptr=stream_ptr
    )
    return tri2nrm

def bwd_tri2normal(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor, dw_tri2nrm: torch.Tensor):
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    device = tri2vtx.device
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    assert tri2vtx.dtype == torch.uint32
    assert tri2vtx.is_contiguous()
    assert tri2vtx.requires_grad == False
    #
    assert len(vtx2xyz.shape) == 2
    assert vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == torch.float32
    assert vtx2xyz.device == device
    assert vtx2xyz.is_contiguous()
    assert vtx2xyz.requires_grad == False
    #
    assert len(dw_tri2nrm.shape) == 2
    assert dw_tri2nrm.shape[1] == 3
    assert dw_tri2nrm.dtype == torch.float32
    assert dw_tri2nrm.device == device
    assert dw_tri2nrm.is_contiguous()
    assert dw_tri2nrm.requires_grad == False
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
        #print(device, stream_ptr)
    dw_vtx2xyz = torch.empty(size=(num_vtx,3), dtype=torch.float32, device=device)
    from .. import TriMesh3
    TriMesh3.bwd_tri2normal(
        tri2vtx.__dlpack__(stream=stream_ptr),
        vtx2xyz.__dlpack__(stream=stream_ptr),
        dw_tri2nrm.__dlpack__(stream=stream_ptr),
        dw_vtx2xyz.__dlpack__(stream=stream_ptr),
        stream_ptr = stream_ptr
    )
    return dw_vtx2xyz


class Tri2Normal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tri2vtx, vtx2xyz):
        ctx.save_for_backward(tri2vtx, vtx2xyz)
        return tri2normal(tri2vtx.detach(), vtx2xyz.detach())

    @staticmethod
    def backward(ctx, dw_tri2nrm):
        tri2vtx, vtx2xyz = ctx.saved_tensors
        dw_vtx2xyz = bwd_tri2normal(tri2vtx.detach(), vtx2xyz.detach(), dw_tri2nrm)
        return None, dw_vtx2xyz

