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
