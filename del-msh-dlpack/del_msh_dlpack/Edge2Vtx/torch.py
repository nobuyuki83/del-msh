import torch
#
from .. import util_torch

def from_vtx2vtx(vtx2idx_offset: torch.Tensor, idx2vtx: torch.Tensor, edge2vtx: torch.Tensor):
    num_vtx = vtx2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_edge = edge2vtx.shape[0]
    device = vtx2idx_offset.device
    #
    assert num_edge == num_idx
    util_torch.assert_shape_dtype_device(vtx2idx_offset, (num_vtx + 1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(edge2vtx, (num_edge, 2), torch.uint32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import Edge2Vtx

    Edge2Vtx.from_vtx2vtx(
        vtx2idx_offset.__dlpack__(),
        idx2vtx.__dlpack__(),
        edge2vtx.__dlpack__(),
        stream_ptr
    )

def contour_for_triangle_mesh(
        tri2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        transform_world2ndc: torch.Tensor,
        edge2vtx: torch.Tensor,
        edge2tri: torch.Tensor) -> torch.Tensor:
    vtx2xyz = vtx2xyz.detach()
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_edge = edge2vtx.shape[0]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_world2ndc, (4, 4), torch.float32, device)
    util_torch.assert_shape_dtype_device(edge2vtx, (num_edge, 2), torch.uint32, device)
    util_torch.assert_shape_dtype_device(edge2tri, (num_edge, 2), torch.uint32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import Edge2Vtx, _CapsuleAsDLPack

    cap_cedge2vtx = Edge2Vtx.contour_for_triangle_mesh(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        transform_world2ndc.T.contiguous().__dlpack__(),
        edge2vtx.__dlpack__(),
        edge2tri.__dlpack__(),
        stream_ptr=stream_ptr
    )
    return torch.from_dlpack(_CapsuleAsDLPack(cap_cedge2vtx)).clone()

