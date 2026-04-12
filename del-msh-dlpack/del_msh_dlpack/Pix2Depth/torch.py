import torch
from .. import util_torch


def update(
        pix2depth: torch.Tensor,
        pix2tri: torch.Tensor,
        tri2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        transform_ndc2world: torch.Tensor):
    """Compute depth image from a pre-computed pix2tri map.

    Args:
        pix2depth: (img_h, img_w) float32 - depth values per pixel, modified in-place
        pix2tri: (img_h, img_w) uint32 - triangle index per pixel (u32::MAX if background)
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions in world space
        transform_ndc2world: (4, 4) float32 - NDC-to-world transformation matrix
    """
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    img_h, img_w = pix2depth.shape[0], pix2depth.shape[1]
    device = pix2depth.device
    #
    util_torch.assert_shape_dtype_device(pix2depth, (img_h, img_w), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_ndc2world, (4, 4), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import Pix2Depth

    Pix2Depth.update(
        pix2depth.detach().__dlpack__(),
        pix2tri.__dlpack__(),
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        transform_ndc2world.T.contiguous().__dlpack__(),
        stream_ptr=stream_ptr,
    )


def bwd_wrt_vtx2xyz(
        dldw_vtx2xyz: torch.Tensor,
        pix2tri: torch.Tensor,
        tri2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        dldw_pix2depth: torch.Tensor,
        transform_ndc2world: torch.Tensor):
    """Backward pass: accumulate gradients of depth w.r.t. vertex positions.

    Args:
        dldw_vtx2xyz: (num_vtx, 3) float32 - gradient accumulator, modified in-place
        pix2tri: (img_h, img_w) uint32 - triangle index per pixel (u32::MAX if background)
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions in world space
        dldw_pix2depth: (img_h, img_w) float32 - loss gradient w.r.t. each pixel depth
        transform_ndc2world: (4, 4) float32 - NDC-to-world transformation matrix
    """
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    img_h, img_w = pix2tri.shape[0], pix2tri.shape[1]
    device = dldw_vtx2xyz.device
    #
    util_torch.assert_shape_dtype_device(dldw_vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(dldw_pix2depth, (img_h, img_w), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_ndc2world, (4, 4), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import Pix2Depth

    Pix2Depth.bwd_wrt_vtx2xyz(
        dldw_vtx2xyz.__dlpack__(),
        pix2tri.__dlpack__(),
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        dldw_pix2depth.__dlpack__(),
        transform_ndc2world.T.contiguous().__dlpack__(),
        stream_ptr=stream_ptr,
    )


class Pix2DepthFunction(torch.autograd.Function):
    """Differentiable depth rendering as a torch.autograd.Function."""

    @staticmethod
    def forward(ctx, vtx2xyz, pix2tri, tri2vtx, transform_ndc2world):
        ctx.save_for_backward(pix2tri, tri2vtx, vtx2xyz, transform_ndc2world)
        pix2depth = torch.zeros(pix2tri.shape, dtype=torch.float32, device=vtx2xyz.device)
        update(pix2depth, pix2tri, tri2vtx, vtx2xyz.detach(), transform_ndc2world)
        return pix2depth

    @staticmethod
    def backward(ctx, dldw_pix2depth):
        pix2tri, tri2vtx, vtx2xyz, transform_ndc2world = ctx.saved_tensors
        dldw_vtx2xyz = torch.zeros_like(vtx2xyz)
        bwd_wrt_vtx2xyz(
            dldw_vtx2xyz,
            pix2tri,
            tri2vtx,
            vtx2xyz.detach(),
            dldw_pix2depth.contiguous(),
            transform_ndc2world,
        )
        return dldw_vtx2xyz, None, None, None
