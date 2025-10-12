import torch
import del_msh_dlpack.Mortons.pt
import del_msh_dlpack.Array1D.pt


def test_01():
    # test 2D
    vtx2co = torch.rand((5, 2))
    transform_co2unit = torch.eye(3)
    vtx2morton = del_msh_dlpack.Mortons.pt.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    print("vtx2morton", vtx2morton)
    (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.pt.argsort(vtx2morton)
    bvhnodes = del_msh_dlpack.Mortons.pt.make_bvh(idx2vtx, idx2morton)

    if torch.cuda.is_available():
        d_vtx2co = vtx2co.cuda()
        d_transform_co2unit = transform_co2unit.cuda()
        d_vtx2morton = del_msh_dlpack.Mortons.pt.vtx2morton_from_vtx2co(
            d_vtx2co, d_transform_co2unit
        )
        assert torch.equal(d_vtx2morton.cpu(), vtx2morton)
        (d_idx2vtx, d_idx2morton) = del_msh_dlpack.Array1D.pt.argsort(d_vtx2morton)
        print("d_vtx2morton", d_vtx2morton)
        assert torch.equal(d_idx2vtx.cpu(), idx2vtx)
        assert torch.equal(d_idx2morton.cpu(), idx2morton)


def test_02():
    # test 3D
    vtx2co = torch.rand((10, 3))
    transform_co2unit = torch.eye(4)
    vtx2morton = del_msh_dlpack.Mortons.pt.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.pt.argsort(vtx2morton)
    bvhnodes = del_msh_dlpack.Mortons.pt.make_bvh(idx2vtx, idx2morton)

    if torch.cuda.is_available():
        d_vtx2co = vtx2co.cuda()
        d_transform_co2unit = transform_co2unit.cuda()
        d_vtx2morton = del_msh_dlpack.Mortons.pt.vtx2morton_from_vtx2co(
            d_vtx2co, d_transform_co2unit
        )
        assert torch.equal(d_vtx2morton.cpu(), vtx2morton)
        (d_idx2vtx, d_idx2morton) = del_msh_dlpack.Array1D.pt.argsort(d_vtx2morton)
        assert torch.equal(d_idx2vtx.cpu(), idx2vtx)
        assert torch.equal(d_idx2morton.cpu(), idx2morton)
