import torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch


def test_01():
    # test 2D
    vtx2co = torch.rand((100, 2))
    transform_co2unit = torch.eye(3)
    vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    # print("vtx2morton", vtx2morton)
    (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
    bvhnodes = del_msh_dlpack.Mortons.torch.make_bvh(idx2vtx, idx2morton)

    if torch.cuda.is_available():
        d_vtx2co = vtx2co.cuda()
        d_transform_co2unit = transform_co2unit.cuda()
        d_vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
            d_vtx2co, d_transform_co2unit
        )
        assert torch.equal(d_vtx2morton.cpu(), vtx2morton)
        (d_idx2vtx, d_idx2morton) = del_msh_dlpack.Array1D.torch.argsort(d_vtx2morton)
        #print("d_vtx2morton", d_vtx2morton)
        assert torch.equal(d_idx2vtx.cpu(), idx2vtx)
        assert torch.equal(d_idx2morton.cpu(), idx2morton)
        d_bvhnodes = del_msh_dlpack.Mortons.torch.make_bvh(idx2vtx, idx2morton)
        #print("bvhnodes", bvhnodes)
        #print("d_bvhnodes", d_bvhnodes)
        assert torch.equal(d_bvhnodes.cpu(), bvhnodes)


def test_02():
    # test 3D
    vtx2co = torch.rand((100, 3))
    transform_co2unit = torch.eye(4)
    vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
    bvhnodes = del_msh_dlpack.Mortons.torch.make_bvh(idx2vtx, idx2morton)

    if torch.cuda.is_available():
        d_vtx2co = vtx2co.cuda()
        d_transform_co2unit = transform_co2unit.cuda()
        d_vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
            d_vtx2co, d_transform_co2unit
        )
        assert torch.equal(d_vtx2morton.cpu(), vtx2morton)
        (d_idx2vtx, d_idx2morton) = del_msh_dlpack.Array1D.torch.argsort(d_vtx2morton)
        assert torch.equal(d_idx2vtx.cpu(), idx2vtx)
        assert torch.equal(d_idx2morton.cpu(), idx2morton)
        d_bvhnodes = del_msh_dlpack.Mortons.torch.make_bvh(idx2vtx, idx2morton)
        #print("bvhnodes", bvhnodes)
        #print("d_bvhnodes", d_bvhnodes)
        assert torch.equal(d_bvhnodes.cpu(), bvhnodes)


def test_03():

    num_dim = 3
    torch.manual_seed(0)
    num_vtx = 1_000
    vtx2xyz = torch.rand((num_vtx, num_dim)) * 3.0 - 1.0
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3), dtype=torch.float32)
    #
    vtx2lhs0 = del_msh_dlpack.NBody.torch.screened_poisson(vtx2xyz, vtx2rhs, 0.1, 0.0001, vtx2xyz)
    #
    transform_world2unit = torch.tensor([
        [1./3., 0., 0., 0.],
        [0., 1./3., 0., 0.],
        [0., 0., 1./3., 0.],
        [0., 0., 0., 1.]]) @ torch.tensor([
        [1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1]
    ])

    vtx2morton0 = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2xyz, transform_world2unit)

    ones = torch.ones((vtx2xyz.shape[0], 1), dtype=vtx2xyz.dtype, device=vtx2xyz.device)
    vtx2xyzw = torch.cat([vtx2xyz, ones], dim=1)  # (N,4)
    vtx2unit = (vtx2xyzw @ transform_world2unit.T)[:,0:3].clone()
    vtx2morton1 = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2unit, torch.eye(4))

    assert torch.equal(vtx2morton0, vtx2morton1)
