import torch


from ..util_torch import assert_shape_dtype_device, to_dlpack_safe


def bwd(
    tri2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor,
    transform_world2pix: torch.Tensor,
    pix2tri: torch.Tensor,
    pix2val: torch.Tensor,
    dldw_pix2val: torch.Tensor,
) -> torch.Tensor:
    """Compute gradient of rasterized edge w.r.t. vertex positions.

    Args:
        :param tri2vtx: (num_tri, 3) uint32
        :param vtx2xyz: (num_vtx, 3) float32
        :param transform_world2pix: (4, 4) float32 - world-to-pixel transform
        :param pix2tri: (H, W) uint32 - triangle index per pixel
        :param dldw_pix2val:
        :param pix2val:

    Returns:
        dldw_vtx2xyz: (num_vtx, 3) float32 - loss gradient w.r.t. each vertex position

    """
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    img_w = pix2val.shape[1]
    img_h = pix2val.shape[0]
    num_vdim = pix2val.shape[2]
    device = tri2vtx.device
    #
    assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    assert_shape_dtype_device(pix2val, (img_h, img_w, num_vdim), torch.float32, device)
    assert_shape_dtype_device(
        dldw_pix2val, (img_h, img_w, num_vdim), torch.float32, device
    )
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import RasterizedEdgeGradient, util_torch

    dldw_vtx2xyz = torch.zeros_like(vtx2xyz)
    RasterizedEdgeGradient.bwd(
        util_torch.to_dlpack_safe(tri2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(dldw_vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(
            transform_world2pix.T.contiguous().flatten(), stream_ptr
        ),
        util_torch.to_dlpack_safe(pix2tri, stream_ptr),
        util_torch.to_dlpack_safe(pix2val, stream_ptr),
        util_torch.to_dlpack_safe(dldw_pix2val, stream_ptr),
        stream_ptr,
    )
    return dldw_vtx2xyz


def edge_gradient_and_type(
    tri2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor,
    transform_world2pix: torch.Tensor,
    pix2tri: torch.Tensor,
    pix2val: torch.Tensor,
    dldw_pix2val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-edge gradient and type for the rasterized image.

    Args:
        tri2vtx: (num_tri, 3) uint32
        vtx2xyz: (num_vtx, 3) float32
        transform_world2pix: (4, 4) float32 - world-to-pixel transform
        pix2tri: (H, W) uint32 - triangle index per pixel
        pix2val:  - value at the pixel
        dldw_pix2val: (H, W) float32 - loss gradient w.r.t. pixel values
    Returns:
        hedge2type: (H-1, W) uint8  - horizontal edge types
        hedge2dldr: (H-1, W) float32 - horizontal edge loss gradients
        vedge2type: (H, W-1) uint8  - vertical edge types
        vedge2dldr: (H, W-1) float32 - vertical edge loss gradients
    """
    vtx2xyz = vtx2xyz.detach()
    pix2val = pix2val.detach()
    #
    assert len(tri2vtx.shape) == 2
    assert len(vtx2xyz.shape) == 2
    assert len(dldw_pix2val.shape) == 3
    #
    img_h, img_w = pix2tri.shape
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_vdim = dldw_pix2val.shape[2]
    device = tri2vtx.device
    #
    assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    assert_shape_dtype_device(transform_world2pix, (4, 4), torch.float32, device)
    assert_shape_dtype_device(pix2tri, (img_h, img_w), torch.uint32, device)
    assert_shape_dtype_device(pix2val, (img_h, img_w, num_vdim), torch.float32, device)
    assert_shape_dtype_device(
        dldw_pix2val, (img_h, img_w, num_vdim), torch.float32, device
    )
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    hedge2type = torch.zeros((img_h - 1, img_w), dtype=torch.uint8, device=device)
    hedge2dldr = torch.zeros((img_h - 1, img_w), dtype=torch.float32, device=device)
    vedge2type = torch.zeros((img_h, img_w - 1), dtype=torch.uint8, device=device)
    vedge2dldr = torch.zeros((img_h, img_w - 1), dtype=torch.float32, device=device)
    #
    from .. import RasterizedEdgeGradient

    RasterizedEdgeGradient.edge_gradient_and_type(
        to_dlpack_safe(tri2vtx, stream_ptr),
        to_dlpack_safe(vtx2xyz, stream_ptr),
        to_dlpack_safe(transform_world2pix.T.contiguous().flatten(), stream_ptr),
        to_dlpack_safe(pix2tri, stream_ptr),
        to_dlpack_safe(pix2val, stream_ptr),
        to_dlpack_safe(dldw_pix2val, stream_ptr),
        to_dlpack_safe(hedge2type, stream_ptr),
        to_dlpack_safe(hedge2dldr, stream_ptr),
        to_dlpack_safe(vedge2type, stream_ptr),
        to_dlpack_safe(vedge2dldr, stream_ptr),
        stream_ptr,
    )
    return hedge2type, hedge2dldr, vedge2type, vedge2dldr


def smooth_gradient(
    hedge2type: torch.Tensor,
    vedge2type: torch.Tensor,
    num_iter: int,
    hedge2dldr: torch.Tensor,
    vedge2dldr: torch.Tensor,
):
    """Smooth staggered-grid edge gradients in-place (100 iterations).

    Args:
        hedge2type: (H-1, W) uint8
        hedge2dldr: (H-1, W) float32
        vedge2type: (H, W-1) uint8
        vedge2dldr: (H, W-1) float32
        num_iter: number of iteration
    """
    device = hedge2type.device
    img_h = hedge2type.shape[0] + 1
    img_w = hedge2type.shape[1]
    #
    assert_shape_dtype_device(hedge2type, (img_h - 1, img_w), torch.uint8, device)
    assert_shape_dtype_device(hedge2dldr, (img_h - 1, img_w), torch.float32, device)
    assert_shape_dtype_device(vedge2type, (img_h, img_w - 1), torch.uint8, device)
    assert_shape_dtype_device(vedge2dldr, (img_h, img_w - 1), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import RasterizedEdgeGradient, util_torch

    RasterizedEdgeGradient.smooth_gradient(
        util_torch.to_dlpack_safe(hedge2type, stream_ptr),
        util_torch.to_dlpack_safe(hedge2dldr, stream_ptr),
        util_torch.to_dlpack_safe(vedge2type, stream_ptr),
        util_torch.to_dlpack_safe(vedge2dldr, stream_ptr),
        num_iter,
        stream_ptr,
    )


def interpolate(hedge2vy: torch.Tensor, vedge2vx: torch.Tensor, vtx2xy: torch.Tensor):
    img_shape = (hedge2vy.shape[1], hedge2vy.shape[0] + 1)
    num_vtx = vtx2xy.shape[0]
    device = hedge2vy.device
    #
    assert_shape_dtype_device(
        hedge2vy, (img_shape[1] - 1, img_shape[0]), dtype=torch.float32, device=device
    )
    assert_shape_dtype_device(
        vedge2vx, (img_shape[1], img_shape[0] - 1), dtype=torch.float32, device=device
    )
    assert_shape_dtype_device(vtx2xy, (num_vtx, 2), dtype=torch.float32, device=device)
    #
    vtx2velo = torch.empty(size=(num_vtx, 2), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import RasterizedEdgeGradient, util_torch

    RasterizedEdgeGradient.interpolate(
        util_torch.to_dlpack_safe(hedge2vy, stream_ptr),
        util_torch.to_dlpack_safe(vedge2vx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xy, stream_ptr),
        util_torch.to_dlpack_safe(vtx2velo, stream_ptr),
        stream_ptr,
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
        dldw_vtx2xyz = bwd(
            tri2vtx,
            vtx2xyz.detach(),
            transform_world2pix,
            pix2tri,
            pix2vin.detach(),
            dldw_pix2vout,
        )
        dldw_pix2vin = dldw_pix2vout.clone()
        return None, dldw_vtx2xyz, None, None, dldw_pix2vin


class AutogradWithSmooth(torch.autograd.Function):
    """rasterized edge gradient as a torch.autograd.Function."""

    @staticmethod
    def forward(ctx, tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2val):
        ctx.save_for_backward(tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2val)
        return pix2val

    @staticmethod
    def backward(ctx, dldw_pix2val):
        tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2val = ctx.saved_tensors
        hedge2type, hedge2dldr, vedge2type, vedge2dldr = edge_gradient_and_type(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2val, dldw_pix2val
        )
        smooth_gradient(hedge2type, vedge2type, 100, hedge2dldr, vedge2dldr)
        #
        """
        import pathlib
        from ..IoVtk.torch import write_velocity_on_staggered_grid
        path0 = pathlib.Path(__file__).parent.parent.parent.parent / "target" / "del_msh_dlpack__microedge2.vtk"
        write_velocity_on_staggered_grid(path0, hedge2dldr, vedge2dldr)
        """
        #
        # Project vertices with homogeneous division.  ``transform_affine`` is
        # insufficient here because world-to-pixel may be a perspective matrix.
        ones = torch.ones_like(vtx2xyz[:, :1])
        vtx2xyzw = torch.cat([vtx2xyz, ones], dim=1) @ transform_world2pix.T
        vtx2wh = vtx2xyzw[:, 0:2] / vtx2xyzw[:, 3:4]
        dldw_vtx2wh = interpolate(hedge2dldr, vedge2dldr, vtx2wh)
        # Chain through q.xy / q.w, then through q = M @ [xyz, 1].
        qx, qy, qw = vtx2xyzw[:, 0], vtx2xyzw[:, 1], vtx2xyzw[:, 3]
        gx, gy = dldw_vtx2wh[:, 0], dldw_vtx2wh[:, 1]
        dldw_q = torch.zeros_like(vtx2xyzw)
        dldw_q[:, 0] = gx / qw
        dldw_q[:, 1] = gy / qw
        dldw_q[:, 3] = -(gx * qx + gy * qy) / (qw * qw)
        dldw_vtx2xyz = (dldw_q @ transform_world2pix)[:, 0:3].clone()
        return None, dldw_vtx2xyz, None, None, dldw_pix2val
