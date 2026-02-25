import numpy as np


def normalize_to_unit_cube(vtx2xyz: np.ndarray) -> np.ndarray:
    """fit vertices into a unit cube [-0.5, 0.5]^3 with uniform scale,
    returned as a 4x4 affine matrix (row-major, applied as p' = mat @ [p, 1]^T)

    Args:
        vtx2xyz: (num_vtx, 3) float32
    Returns:
        mat: (4, 4) float32
    """
    assert vtx2xyz.shape[1] == 3 and vtx2xyz.dtype == np.float32
    aabb_min = vtx2xyz.min(axis=0)  # (3,)
    aabb_max = vtx2xyz.max(axis=0)  # (3,)
    center = (aabb_min + aabb_max) * 0.5
    s = 1.0 / (aabb_max - aabb_min).max()
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = s
    mat[1, 1] = s
    mat[2, 2] = s
    mat[3, 3] = 1.0
    mat[:3, 3] = -center * s
    return mat
