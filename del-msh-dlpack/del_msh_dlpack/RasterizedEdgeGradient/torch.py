import torch


def gradient(
    tri2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor,
    transform_world2pix: torch.Tensor,
    dldw_pixval: torch.Tensor,
    pix2tri: torch.Tensor,
) -> torch.Tensor:
    """Compute gradient of rasterized edge w.r.t. vertex positions.

    Args:
        tri2vtx: (num_tri, 3) uint32
        vtx2xyz: (num_vtx, 3) float32
        transform_world2pix: (4, 4) or (16,) float32 - world-to-pixel transform
        dldw_pixval: (H, W) float32 - loss gradient w.r.t. pixel values
        pix2tri: (H, W) uint32 - triangle index per pixel

    Returns:
        dldw_vtx2xyz: (num_vtx, 3) float32 - loss gradient w.r.t. each vertex position
    """
    from .. import RasterizedEdgeGradient, util_torch
    dldw_vtx2xyz = torch.zeros_like(vtx2xyz)
    RasterizedEdgeGradient.bwd(
        util_torch.to_dlpack_safe(tri2vtx, 0),
        util_torch.to_dlpack_safe(vtx2xyz, 0),
        util_torch.to_dlpack_safe(dldw_vtx2xyz, 0),
        util_torch.to_dlpack_safe(transform_world2pix.T.contiguous().flatten(), 0),
        util_torch.to_dlpack_safe(dldw_pixval, 0),
        util_torch.to_dlpack_safe(pix2tri, 0),
    )
    return dldw_vtx2xyz


def edge_gradient_and_type(
    tri2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor,
    transform_world2pix: torch.Tensor,
    dldw_pixval: torch.Tensor,
    pix2tri: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-edge gradient and type for the rasterized image.

    Args:
        tri2vtx: (num_tri, 3) uint32
        vtx2xyz: (num_vtx, 3) float32
        transform_world2pix: (4, 4) or (16,) float32 - world-to-pixel transform
        dldw_pixval: (H, W) float32 - loss gradient w.r.t. pixel values
        pix2tri: (H, W) uint32 - triangle index per pixel
    Returns:
        hedge2type: (H-1, W) uint8  - horizontal edge types
        hedge2dldr: (H-1, W) float32 - horizontal edge loss gradients
        vedge2type: (H, W-1) uint8  - vertical edge types
        vedge2dldr: (H, W-1) float32 - vertical edge loss gradients
    """
    from .. import RasterizedEdgeGradient, util_torch
    img_h, img_w = pix2tri.shape
    hedge2type = torch.zeros((img_h - 1, img_w), dtype=torch.uint8)
    hedge2dldr = torch.zeros((img_h - 1, img_w), dtype=torch.float32)
    vedge2type = torch.zeros((img_h, img_w - 1), dtype=torch.uint8)
    vedge2dldr = torch.zeros((img_h, img_w - 1), dtype=torch.float32)
    transform_flat = transform_world2pix.T.contiguous().flatten()
    RasterizedEdgeGradient.edge_gradient_and_type(
        util_torch.to_dlpack_safe(tri2vtx, 0),
        util_torch.to_dlpack_safe(vtx2xyz, 0),
        util_torch.to_dlpack_safe(transform_flat, 0),
        util_torch.to_dlpack_safe(dldw_pixval, 0),
        util_torch.to_dlpack_safe(pix2tri, 0),
        util_torch.to_dlpack_safe(hedge2type, 0),
        util_torch.to_dlpack_safe(hedge2dldr, 0),
        util_torch.to_dlpack_safe(vedge2type, 0),
        util_torch.to_dlpack_safe(vedge2dldr, 0),
    )
    return hedge2type, hedge2dldr, vedge2type, vedge2dldr


def smooth_gradient(
    hedge2type: torch.Tensor,
    hedge2dldr: torch.Tensor,
    vedge2type: torch.Tensor,
    vedge2dldr: torch.Tensor,
):
    """Smooth staggered-grid edge gradients in-place (100 iterations).

    Args:
        hedge2type: (H-1, W) uint8
        hedge2dldr: (H-1, W) float32
        vedge2type: (H, W-1) uint8
        vedge2dldr: (H, W-1) float32
    """
    from .. import RasterizedEdgeGradient, util_torch
    RasterizedEdgeGradient.smooth_gradient(
        util_torch.to_dlpack_safe(hedge2type, 0),
        util_torch.to_dlpack_safe(hedge2dldr, 0),
        util_torch.to_dlpack_safe(vedge2type, 0),
        util_torch.to_dlpack_safe(vedge2dldr, 0),
    )
