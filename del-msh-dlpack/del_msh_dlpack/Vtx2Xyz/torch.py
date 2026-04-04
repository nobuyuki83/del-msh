import torch


def make_mat44_from_fit_into_unit_cube(vtx2xyz: torch.Tensor) -> torch.Tensor:
    """fit vertices into a unit cube [-0.5, 0.5]^3 with uniform scale,
    returned as a 4x4 affine matrix (row-major, applied as p' = mat @ [p, 1]^T)

    Args:
        vtx2xyz: (num_vtx, 3) float32
    Returns:
        mat: (4, 4) float32
    """
    assert vtx2xyz.shape[1] == 3 and vtx2xyz.dtype == torch.float32
    aabb_min = vtx2xyz.min(dim=0).values  # (3,)
    aabb_max = vtx2xyz.max(dim=0).values  # (3,)
    center = (aabb_min + aabb_max) * 0.5
    s = 1.0 / (aabb_max - aabb_min).max()
    mat = torch.zeros(4, 4, dtype=torch.float32, device=vtx2xyz.device)
    mat[0, 0] = s
    mat[1, 1] = s
    mat[2, 2] = s
    mat[3, 3] = 1.0
    mat[:3, 3] = -center * s + 0.5
    return mat


def transform_affine(vtx2xyz, transform):
    """Apply a 4x4 affine transformation to a set of 3D points."""
    ones = torch.ones((vtx2xyz.shape[0], 1), dtype=torch.float, device=vtx2xyz.device)
    vtx2xyzw = torch.cat([vtx2xyz, ones], dim=1)  # (N,4)
    return (vtx2xyzw @ transform.T)[:, 0:3].clone()