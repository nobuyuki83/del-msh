import torch
from .. import util_torch


def antialias(
        cedge2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        transform_world2pix: torch.Tensor,
        pix2tri: torch.Tensor,
        pix2vin: torch.Tensor) -> torch.Tensor:
    """Apply anti-aliasing to a rendered triangle mesh image by blending pixel
    values along silhouette (contour) edges.

    For each contour edge, pixels that straddle the edge boundary are given
    fractional coverage values (between 0 and 1) based on where the edge
    crosses the pixel boundary.

    Args:
        cedge2vtx: (num_contour, 2) uint32 - contour edge connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions in world space
        transform_world2pix: (4, 4) float32 - world-to-pixel transformation matrix
        pix2tri: (img_h, img_w) uint32 - triangle index per pixel (u32::MAX if background)
        pix2vin: (img_h, img_w) float32 - image to modify in-place
    """
    pix2vin = pix2vin.detach()
    num_contour = cedge2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    img_h, img_w = pix2vin.shape[0], pix2vin.shape[1]
    device = cedge2vtx.device
    #
    util_torch.assert_shape_dtype_device(cedge2vtx, (num_contour, 2), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_world2pix, (4, 4), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    util_torch.assert_shape_dtype_device(pix2vin, (img_h, img_w), torch.float32, device)
    pix2vout = pix2vin.clone()
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import DifferentiableAntialias

    DifferentiableAntialias.antialias(
        cedge2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        transform_world2pix.T.contiguous().__dlpack__(),
        pix2tri.__dlpack__(),
        pix2vin.__dlpack__(),
        pix2vout.__dlpack__(),
        stream_ptr=stream_ptr
    )
    return pix2vout


def bwd_antialias(
        cedge2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        transform_world2pix: torch.Tensor,
        pix2val: torch.Tensor,
        dldw_pix2val: torch.Tensor,
        pix2tri: torch.Tensor):
    """Backward pass of antialias: compute gradient of vertex positions from
    pixel-space loss gradients along silhouette edges.

    Args:
        cedge2vtx: (num_contour, 2) uint32 - contour edge connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions in world space
        dldw_vtx2xyz: (num_vtx, 3) float32 - gradient accumulator for vertex positions (modified in-place)
        transform_world2pix: (4, 4) float32 - world-to-pixel transformation matrix
        dldw_pix2val: (img_h, img_w) float32 - loss gradient w.r.t. each pixel value
        pix2tri: (img_h, img_w) uint32 - triangle index per pixel (u32::MAX if background)
    """
    pix2val = pix2val.detach()
    vtx2xyz = vtx2xyz.detach()
    num_contour = cedge2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    img_h, img_w = dldw_pix2val.shape[0], dldw_pix2val.shape[1]
    device = cedge2vtx.device
    #
    util_torch.assert_shape_dtype_device(cedge2vtx, (num_contour, 2), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_world2pix, (4, 4), torch.float32, device)
    util_torch.assert_shape_dtype_device(dldw_pix2val, (img_h, img_w), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2val, (img_h, img_w), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    #
    dldw_vtx2xyz = torch.zeros_like(vtx2xyz)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import DifferentiableAntialias

    DifferentiableAntialias.bwd_antialias(
        cedge2vtx.__dlpack__(),
        vtx2xyz.detach().__dlpack__(),
        dldw_vtx2xyz.__dlpack__(),
        transform_world2pix.T.contiguous().__dlpack__(),
        pix2val.detach().__dlpack__(),
        dldw_pix2val.__dlpack__(),
        pix2tri.__dlpack__(),
        stream_ptr=stream_ptr
    )
    return dldw_vtx2xyz


class DifferentiableAntialiasFunction(torch.autograd.Function):
    """Differentiable antialias as a torch.autograd.Function."""

    @staticmethod
    def forward(ctx, cedge2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin):
        ctx.save_for_backward(cedge2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin)
        return antialias(cedge2vtx.detach(), vtx2xyz.detach(), transform_world2pix.detach(), pix2tri.detach(), pix2vin)

    @staticmethod
    def backward(ctx, dldw_pix2vout):
        cedge2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin = ctx.saved_tensors
        dldw_vtx2xyz = bwd_antialias(
            cedge2vtx,
            vtx2xyz,
            transform_world2pix,
            pix2vin,
            dldw_pix2vout,
            pix2tri)
        dldw_pix2vin = dldw_pix2vout.clone()
        return None, dldw_vtx2xyz, None, None, dldw_pix2vin
