import torch

from ..util_torch import assert_shape_dtype_device, to_dlpack_safe

import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz


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
    from .. import EdgeGrad, util_torch

    dldw_vtx2xyz = torch.zeros_like(vtx2xyz)
    EdgeGrad.bwd(
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
