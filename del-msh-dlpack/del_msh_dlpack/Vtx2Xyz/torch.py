import torch


def normalize_to_unit_cube(vtx2xyz: torch.Tensor) -> torch.Tensor:
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
    mat[:3, 3] = -center * s
    return mat
