import math
#
import torch

def from_x_rotation(angle: float, device=None, dtype=torch.float32) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([
        [1,  0,  0,  0],
        [0,  c, -s,  0],
        [0,  s,  c,  0],
        [0,  0,  0,  1],
    ], device=device, dtype=dtype)

def from_translation(tx: float, ty: float, tz: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1],
    ], device=device, dtype=dtype)

def from_scale(sx: float, sy: float, sz: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0,  1],
    ], device=device, dtype=dtype)

'''
transformation converting normalized device coordinate (NDC) `[-1,+1]^3` to pixel coordinate
depth (-1, +1) is transformed to (0, +1)
for example:
     [-1,-1,-1] becomes (0, H, 0)
     [+1,+1,+1] becomes (W, 0, 1)

 * Arguments
    * `image_shape` - (width, height)
'''
def from_transform_ndc2pix(img_shape, device=None, dtype=torch.float32):
    return torch.tensor(
        [
            [
                0.5 * float(img_shape[0]),
                0,
                0,
                0.5 * float(img_shape[0]),
             ],
            [
                0,
                -0.5 * float(img_shape[1]),
                0,
                0.5 * float(img_shape[1]),
            ],
            [
                0,
                0,
                0.5,
                0.5,
            ],
            [
                0,
                0,
                0,
                1,
            ]

        ], device=device, dtype=dtype)

def from_uniform_scale(s, device):
    return torch.tensor([
        [s, 0., 0., 0.],
        [0., s, 0., 0.],
        [0., 0., s, 0.],
        [0., 0., 0., 1.]
    ], device = device)

def from_transfrom_world2unit(vtx2xyz, device):
    xyz_min = vtx2xyz.min(dim=0).values
    xyz_max = vtx2xyz.max(dim=0).values
    xyz_len = xyz_max - xyz_min
    scale = 1.0/xyz_len.max().item()
    xyz_center = (xyz_min + xyz_max)*0.5
    # print(xyz_min, xyz_max, xyz_len, xyz_center, scale)
    m1 = from_translation(-xyz_center[0], -xyz_center[1], -xyz_center[2], device)
    m2 = from_uniform_scale(scale, device)
    m3 = from_translation(0.5, 0.5, 0.5, device)
    return m3 @ m2 @ m1

def from_fit_vtx2xyz_into_unit_cube(vtx2xyz: torch.Tensor) -> torch.Tensor:
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