import torch

'''
def transform_affine(vtx2xyz, transform):
    """Apply a 4x4 affine transformation to a set of 3D points."""
    ones = torch.ones((vtx2xyz.shape[0], 1), dtype=torch.float, device=vtx2xyz.device)
    vtx2xyzw = torch.cat([vtx2xyz, ones], dim=1)  # (N,4)
    return (vtx2xyzw @ transform.T)[:, 0:3].clone()
'''


def transform_homography(vtx2xyz, transform):
    assert len(vtx2xyz.shape) == 2
    assert vtx2xyz.shape[1] == 3
    assert transform.shape == (4, 4)
    #
    ones = torch.ones((vtx2xyz.shape[0], 1), dtype=torch.float, device=vtx2xyz.device)
    vtx2xyzw = torch.cat([vtx2xyz, ones], dim=1)  # (N,4)
    vtx2abcd = vtx2xyzw @ transform.T  # (N,4)
    return (vtx2abcd[:, 0:3] / vtx2abcd[:, 3:4]).clone().contiguous()


def transform_homography_jacobian(vtx2xyz: torch.Tensor, transform: torch.Tensor):
    # Jacobian of the homography transform uvw = (Ax+b)/(cx+d) w.r.t. x.
    # By the quotient rule: d(uvw)/dx = A/(cx+d) - (Ax+b)*c/(cx+d)^2
    # Returns shape (N, 3, 3): jacobian[i, j, k] = d(uvw[i,j])/d(xyz[i,k])
    assert transform.shape == (4, 4)

    a = transform[:3, :3]  # (3, 3)
    c = transform[3, :3]  # (3,) — homogeneous row

    axb = transform[:3, :3] @ vtx2xyz.T + transform[:3, 3:4]  # (3, N): numerator Ax+b
    cxd = transform[3:4, :3] @ vtx2xyz.T + transform[3, 3]  # (1, N): denominator cx+d

    vtx2duvw2dxyz = (
        a[None, :, :] / cxd.T[:, None, :]
        - axb.T[:, :, None] * c[None, None, :] / cxd.T[:, None, :].square()
    )

    return vtx2duvw2dxyz.contiguous()  # (N, 3, 3)


def save_wavefront_obj(xyz: torch.Tensor, path_dir: str):
    from del_msh_dlpack.IoWavefrontObj.torch import save_points
    save_points(xyz, path_dir)


def from_sample_aabb(aabb, num_sample):
    assert aabb.shape == (2, 3)
    u = torch.rand((num_sample, 3), device=aabb.device, dtype=aabb.dtype)
    return aabb[0] + (aabb[1] - aabb[0]) * u

def normalize(vtx2xyz: torch.Tensor, size: float = 1.0) -> torch.Tensor:
    """Center and uniformly scale vertices so the longest axis fits in [-size/2, size/2]."""
    aabb_min = vtx2xyz.min(dim=0).values
    aabb_max = vtx2xyz.max(dim=0).values
    center = (aabb_min + aabb_max) * 0.5
    scale = size / (aabb_max - aabb_min).max()
    return (vtx2xyz - center) * scale
