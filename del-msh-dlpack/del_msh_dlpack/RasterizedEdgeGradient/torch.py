import torch
#

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


def interpolate(
    hedge2vy: torch.Tensor,
    vedge2vx: torch.Tensor,
    vtx2xy: torch.Tensor):
    img_shape = (hedge2vy.shape[0]+1, hedge2vy.shape[1])
    assert vedge2vx.shape == (img_shape[0],img_shape[1]-1)
    num_vtx = vtx2xy.shape[0]
    vtx2velo = torch.empty(size=(num_vtx, 2), dtype=torch.float32)
    from .. import RasterizedEdgeGradient, util_torch
    RasterizedEdgeGradient.interpolate(
        util_torch.to_dlpack_safe(hedge2vy, 0),
        util_torch.to_dlpack_safe(vedge2vx, 0),
        util_torch.to_dlpack_safe(vtx2xy, stream_ptr=0),
        util_torch.to_dlpack_safe(vtx2velo, stream_ptr=0)
    )
    return vtx2velo


class RasterizedEdgeGradientFunction(torch.autograd.Function):
    """rasterized edge gradient as a torch.autograd.Function."""

    @staticmethod
    def forward(ctx, tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin):
        ctx.save_for_backward(tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin)
        return pix2vin

    @staticmethod
    def backward(ctx, dldw_pix2vout):
        tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin = ctx.saved_tensors
        dldw_vtx2xyz = gradient(tri2vtx, vtx2xyz.detach(), transform_world2pix, dldw_pix2vout, pix2tri)
        dldw_pix2vin = dldw_pix2vout.clone()
        return None, dldw_vtx2xyz, None, None, dldw_pix2vin


class RasterizedEdgeGradientWithSmoothFunction(torch.autograd.Function):
    """rasterized edge gradient as a torch.autograd.Function."""

    @staticmethod
    def forward(ctx, tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin):
        ctx.save_for_backward(tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin)
        return pix2vin

    @staticmethod
    def backward(ctx, dldw_pix2vout):
        tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin = ctx.saved_tensors
        hedge2type, hedge2dldr, vedge2type, vedge2dldr = edge_gradient_and_type(
            tri2vtx, vtx2xyz.detach(), transform_world2pix,dldw_pix2vout,pix2tri)
        smooth_gradient(hedge2type, hedge2dldr, vedge2type, vedge2dldr)
        from del_msh_dlpack.Vtx2Xyz.torch import transform_affine
        vtx2wh = transform_affine(vtx2xyz, transform_world2pix)[:,0:2].clone()
        dldw_vtx2wh = interpolate(hedge2dldr, vedge2dldr, vtx2wh)
        zeros = torch.zeros((dldw_vtx2wh.shape[0], 1), dtype=torch.float, device=dldw_vtx2wh.device)
        dldw_vtx2whdw = torch.cat([dldw_vtx2wh, zeros, zeros], dim=1)  # (N,4)
        dldw_vtx2xyz = (dldw_vtx2whdw @ transform_world2pix.clone() )[:, 0:3].clone()
        dldw_pix2vin = dldw_pix2vout.clone()
        return None, dldw_vtx2xyz, None, None, dldw_pix2vin