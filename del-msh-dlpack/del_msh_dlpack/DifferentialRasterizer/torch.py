import torch
from .. import util_torch


def antialias(
        edge2vtx_contour: torch.Tensor,
        vtx2xyz: torch.Tensor,
        transform_world2pix: torch.Tensor,
        pix2tri: torch.Tensor,
        img_data: torch.Tensor
    ):
    """Apply anti-aliasing to a rendered triangle mesh image by blending pixel
    values along silhouette (contour) edges.

    For each contour edge, pixels that straddle the edge boundary are given
    fractional coverage values (between 0 and 1) based on where the edge
    crosses the pixel boundary.

    Args:
        edge2vtx_contour: (num_contour, 2) uint32 - contour edge connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions in world space
        transform_world2pix: (4, 4) float32 - world-to-pixel transformation matrix
        pix2tri: (img_h, img_w) uint32 - triangle index per pixel (u32::MAX if background)
        img_data: (img_h, img_w) float32 - image to modify in-place
    """
    num_contour = edge2vtx_contour.shape[0]
    num_vtx = vtx2xyz.shape[0]
    img_h, img_w = img_data.shape[0], img_data.shape[1]
    device = edge2vtx_contour.device
    #
    util_torch.assert_shape_dtype_device(edge2vtx_contour, (num_contour, 2), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_world2pix, (4, 4), torch.float32, device)
    util_torch.assert_shape_dtype_device(img_data, (img_h, img_w), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    #
    from .. import DifferentialRasterizer

    DifferentialRasterizer.antialias(
        edge2vtx_contour.__dlpack__(),
        vtx2xyz.__dlpack__(),
        transform_world2pix.T.contiguous().__dlpack__(),
        pix2tri.__dlpack__(),
        img_data.__dlpack__(),
    )


def bwd_antialias(
        edge2vtx_contour: torch.Tensor,
        vtx2xyz: torch.Tensor,
        dldw_vtx2xyz: torch.Tensor,
        transform_world2pix: torch.Tensor,
        dldw_pixval: torch.Tensor,
        pix2tri: torch.Tensor):
    """Backward pass of antialias: compute gradient of vertex positions from
    pixel-space loss gradients along silhouette edges.

    Args:
        edge2vtx_contour: (num_contour, 2) uint32 - contour edge connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions in world space
        dldw_vtx2xyz: (num_vtx, 3) float32 - gradient accumulator for vertex positions (modified in-place)
        transform_world2pix: (4, 4) float32 - world-to-pixel transformation matrix
        dldw_pixval: (img_h, img_w) float32 - loss gradient w.r.t. each pixel value
        pix2tri: (img_h, img_w) uint32 - triangle index per pixel (u32::MAX if background)
    """
    num_contour = edge2vtx_contour.shape[0]
    num_vtx = vtx2xyz.shape[0]
    img_h, img_w = dldw_pixval.shape[0], dldw_pixval.shape[1]
    device = edge2vtx_contour.device
    #
    util_torch.assert_shape_dtype_device(edge2vtx_contour, (num_contour, 2), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(dldw_vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(transform_world2pix, (4, 4), torch.float32, device)
    util_torch.assert_shape_dtype_device(dldw_pixval, (img_h, img_w), torch.float32, device)
    util_torch.assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    #
    from .. import DifferentialRasterizer

    DifferentialRasterizer.bwd_antialias(
        edge2vtx_contour.__dlpack__(),
        vtx2xyz.__dlpack__(),
        dldw_vtx2xyz.__dlpack__(),
        transform_world2pix.T.contiguous().__dlpack__(),
        dldw_pixval.__dlpack__(),
        pix2tri.__dlpack__(),
    )
