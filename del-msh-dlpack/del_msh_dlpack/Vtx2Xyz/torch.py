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
    print(vtx2xyz.shape)
    return (vtx2abcd[:, 0:3] / vtx2abcd[:, 3:4]).clone().contiguous()


def transform_homography_jacobian(vtx2xyz: torch.Tensor, transform: torch.Tensor):
    assert transform.shape == (4, 4)

    a = transform[:3, :3]  # (3, 3)
    c = transform[3, :3]  # (3,)

    axb = transform[:3, :3] @ vtx2xyz.T + transform[:3, 3:4]  # (3, N)
    cxd = transform[3:4, :3] @ vtx2xyz.T + transform[3, 3]  # (1, N)

    jacobian = (
        a[None, :, :] / cxd.T[:, None, :]
        - axb.T[:, :, None] * c[None, None, :] / cxd.T[:, None, :].square()
    )

    return jacobian.contiguous()  # (N, 3, 3)
