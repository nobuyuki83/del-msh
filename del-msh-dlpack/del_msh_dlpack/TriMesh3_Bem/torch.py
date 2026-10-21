import torch

from .. import util_torch


def eval_kelvinlet_bruteforce(
    tri2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor,
    vtx2rhs: torch.Tensor,
    wtx2xyz: torch.Tensor,
    mu: float,
    nu: float,
) -> torch.Tensor:
    """Evaluate Kelvinlet BEM bruteforce.

    Args:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - source vertex positions
        vtx2rhs: (num_vtx, 3) float32 - source vertex right-hand side values
        wtx2xyz: (num_wtx, 3) float32 - evaluation point positions
        mu: shear modulus
        nu: Poisson's ratio
    Returns:
        wtx2lhs: (num_wtx, 3) float32 - Kelvinlet potential at evaluation points
    """
    wtx2xyz = wtx2xyz.detach()
    vtx2xyz = vtx2xyz.detach()
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_wtx = wtx2xyz.shape[0]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(vtx2rhs, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(wtx2xyz, (num_wtx, 3), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    wtx2lhs = torch.zeros(size=(num_wtx, 3), dtype=torch.float32, device=device)
    from .. import TriMesh3_Bem

    TriMesh3_Bem.eval_kelvinlet_bruteforce(
        util_torch.to_dlpack_safe(tri2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        util_torch.to_dlpack_safe(wtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(wtx2lhs, stream_ptr),
        mu,
        nu,
        stream_ptr=stream_ptr,
    )
    return wtx2lhs
