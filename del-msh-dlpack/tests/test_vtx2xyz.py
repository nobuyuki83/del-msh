import torch

import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz


def test_jacobian_transform():
    torch.manual_seed(0)
    transform = torch.rand((4, 4), dtype=torch.float64)
    vtx2xyz0 = torch.rand((100, 3), dtype=torch.float64)
    #
    vtx2uvw0 = Vtx2Xyz.transform_homography(vtx2xyz0, transform)
    #
    vtx2dxyz = torch.rand((100, 3), dtype=torch.float64)
    eps = 8.0e-5
    vtx2xyz1 = vtx2xyz0 + vtx2dxyz * eps
    vtx2uvw1 = Vtx2Xyz.transform_homography(vtx2xyz1, transform)
    #
    vtx2duvw2dxyz = Vtx2Xyz.transform_homography_jacobian(vtx2xyz0, transform)
    #
    # vtx2dxyz2duvw = vtx2dxyz2duvw.transpose(1,2)
    vtx2duvw_ana = (vtx2duvw2dxyz @ vtx2dxyz.unsqueeze(-1)).squeeze(-1)  # (N, 3)
    #
    vtx2duvw_num = (vtx2uvw1 - vtx2uvw0) / eps
    err = (vtx2duvw_ana - vtx2duvw_num).abs()
    assert err.max().item() < 0.0065
    # print(err.max().item())
    #
    dldw_vtx2uv = torch.rand((vtx2xyz0.shape[0], 2), dtype=torch.float64)
    dldw_vtx2xyz = (dldw_vtx2uv.unsqueeze(1) @ vtx2duvw2dxyz[:, 0:2, :]).squeeze(1)
    l0 = (dldw_vtx2uv * vtx2uvw0[:, 0:2]).sum()
    l1 = (dldw_vtx2uv * vtx2uvw1[:, 0:2]).sum()
    dl_num = (l1 - l0) / eps
    #
    dl_ana = (dldw_vtx2xyz * vtx2dxyz).sum()
    print((dl_num - dl_ana).abs())
    assert (dl_num - dl_ana).abs() < 5.0e-4
