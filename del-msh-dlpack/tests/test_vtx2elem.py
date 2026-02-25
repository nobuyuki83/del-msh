import torch


def test_01():
    """
    test Vtx2Vtx for numpy
    """
    import del_msh_dlpack.Vtx2Elem.torch

    tri2vtx, vtx2xyz = del_msh_dlpack.TriMesh3.torch.torus(1.0, 0.3, 64, 128)
    h_vtx2elem = del_msh_dlpack.Vtx2Elem.torch.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0])
    if torch.cuda.is_available():
        d_vtx2elem = del_msh_dlpack.Vtx2Elem.torch.from_uniform_mesh(
            tri2vtx.cuda(), vtx2xyz.shape[0]
        )
        assert torch.equal(h_vtx2elem[0], d_vtx2elem[0].cpu())
        assert torch.equal(h_vtx2elem[1], d_vtx2elem[1].cpu())
