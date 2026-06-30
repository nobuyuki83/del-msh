import torch

def transform_affine(vtx2xyz, transform):
    """Apply a 4x4 affine transformation to a set of 3D points."""
    ones = torch.ones((vtx2xyz.shape[0], 1), dtype=torch.float, device=vtx2xyz.device)
    vtx2xyzw = torch.cat([vtx2xyz, ones], dim=1)  # (N,4)
    return (vtx2xyzw @ transform.T)[:, 0:3].clone()


def write_vtk_points_with_velocity(path: str, vtx2xyz: torch.Tensor, vtx2velocity: torch.Tensor):
    from .. import util_torch
    from ..del_msh_dlpack import io_vtk_write_points_with_velocity
    assert vtx2xyz.ndim == 2 and vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == torch.float32
    assert vtx2velocity.shape == vtx2xyz.shape
    assert vtx2velocity.dtype == torch.float32
    vtx2xyz = vtx2xyz.contiguous()
    vtx2velocity = vtx2velocity.contiguous()
    io_vtk_write_points_with_velocity(
        path,
        util_torch.to_dlpack_safe(vtx2xyz, 0),
        util_torch.to_dlpack_safe(vtx2velocity, 0),
    )
