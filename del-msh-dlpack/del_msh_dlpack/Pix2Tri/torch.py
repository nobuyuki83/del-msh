import torch
#
from .. import util_torch

def update_pix2tri(
        tri2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        bvhnodes: torch.Tensor,
        bvhnode2aabb: torch.Tensor,
        transform_ndc2world: torch.Tensor,
        pix2tri: torch.Tensor):
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_bvhnode = bvhnodes.shape[0]
    img_w = pix2tri.shape[1]
    img_h = pix2tri.shape[0]
    device = tri2vtx.device
    vtx2xyz = vtx2xyz.detach()
    #
    assert num_bvhnode == num_tri * 2 - 1
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri,3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(bvhnodes, (num_bvhnode, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(bvhnode2aabb, (num_bvhnode, 6), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream

    from ..Pix2Tri import update_pix2tri

    update_pix2tri(
        tri2vtx.__dlpack__(),
        vtx2xyz.detach().__dlpack__(),
        bvhnodes.__dlpack__(),
        bvhnode2aabb.__dlpack__(),
        transform_ndc2world.T.contiguous().__dlpack__(),
        pix2tri.__dlpack__(),
        stream_ptr=stream_ptr,
    )
