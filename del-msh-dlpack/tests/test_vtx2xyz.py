import torch

import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz


def test_jacobian_transform():
    transform = torch.rand((4, 4), dtype=torch.float32)
    vtx2xyz0 = torch.rand((100, 3), dtype=torch.float32)
    #
    vtx2uvw0 = Vtx2Xyz.transform_homography(vtx2xyz0, transform)
    #
    eps = 1.0e-4
    vtx2xyz1 = vtx2xyz0 + torch.rand((100, 3), dtype=torch.float32) * eps
    vtx2uvw1 = Vtx2Xyz.transform_homography(vtx2xyz1, transform)
    #
    Vtx2Xyz.transform_homography_jacobian(vtx2xyz0, transform)
