import torch
#
from .. import util_torch


def by_raycasting(
        tri2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        bvhnodes: torch.Tensor,
        bvhnode2aabb: torch.Tensor,
        transform_ndc2world: torch.Tensor,
        img_shape) -> torch.Tensor:
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_bvhnode = bvhnodes.shape[0]
    device = tri2vtx.device
    vtx2xyz = vtx2xyz.detach()
    transform_ndc2world = transform_ndc2world.contiguous()
    #
    assert num_bvhnode == num_tri * 2 - 1
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(bvhnodes, (num_bvhnode, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(bvhnode2aabb, (num_bvhnode, 6), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_ndc2world, (4, 4), torch.float32, device)
    #
    pix2tri = torch.empty((img_shape[1], img_shape[0]), dtype=torch.uint32, device=device)
    pix2tri.fill_(torch.iinfo(torch.uint32).max)
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

    return pix2tri


def interpolate_fwd(
        pix2tri: torch.Tensor,
        tri2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        vtx2val: torch.Tensor,
        transform_ndc2world: torch.Tensor):
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_vdim = vtx2val.shape[1]
    img_h = pix2tri.shape[0]
    img_w = pix2tri.shape[1]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz.detach(), (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(vtx2val.detach(), (num_vtx, num_vdim), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_ndc2world, (4, 4), torch.float32, device)
    #
    pix2val = torch.zeros((img_h, img_w, num_vdim), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from ..Pix2Tri import interpolate
    interpolate(
        pix2tri.__dlpack__(),
        tri2vtx.__dlpack__(),
        vtx2xyz.detach().__dlpack__(),
        vtx2val.detach().__dlpack__(),
        transform_ndc2world.T.contiguous().__dlpack__(),
        pix2val.__dlpack__(),
        stream_ptr=stream_ptr
    )

    return pix2val


def interpolate_bwd(
        pix2tri: torch.Tensor,
        tri2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        vtx2val: torch.Tensor,
        transform_ndc2world: torch.Tensor,
        dldw_pix2val: torch.Tensor,
        dldw_vtx2xyz: torch.Tensor,
        dldw_vtx2val: torch.Tensor):
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_vdim = vtx2val.shape[1]
    img_h = pix2tri.shape[0]
    img_w = pix2tri.shape[1]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(vtx2val, (num_vtx, num_vdim), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_ndc2world, (4, 4), torch.float32, device)
    util_torch.assert_shape_dtype_device(dldw_pix2val, (img_h, img_w, num_vdim), torch.float32, device)
    util_torch.assert_shape_dtype_device(dldw_vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(dldw_vtx2val, (num_vtx, num_vdim), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from ..Pix2Tri import interpolate_bwd
    interpolate_bwd(
        pix2tri.__dlpack__(),
        tri2vtx.__dlpack__(),
        vtx2xyz.detach().__dlpack__(),
        vtx2val.detach().__dlpack__(),
        transform_ndc2world.T.contiguous().__dlpack__(),
        dldw_pix2val.contiguous().__dlpack__(),
        dldw_vtx2xyz.__dlpack__(),
        dldw_vtx2val.__dlpack__(),
        stream_ptr=stream_ptr
    )


class AutogradInterpolate(torch.autograd.Function):
    """Differentiable depth rendering as a torch.autograd.Function."""

    @staticmethod
    def forward(ctx, pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world):
        ctx.save_for_backward(pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world)
        pix2val = interpolate_fwd(pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world)
        return pix2val

    @staticmethod
    def backward(ctx, dldw_pix2val):
        pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world = ctx.saved_tensors
        dldw_vtx2xyz = torch.zeros_like(vtx2xyz)
        dldw_vtx2val = torch.zeros_like(vtx2val)
        interpolate_bwd(
            pix2tri,
            tri2vtx,
            vtx2xyz.detach(),
            vtx2val.detach(),
            transform_ndc2world,
            dldw_pix2val.contiguous(),
            dldw_vtx2xyz,
            dldw_vtx2val
        )
        return None, None, dldw_vtx2xyz, dldw_vtx2val, None


def interpolate(pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world):
    return AutogradInterpolate.apply(pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world)
