import numpy
import torch

#
import del_msh_dlpack.Vtx2Vtx.numpy


def test_01():
    """
    test Vtx2Vtx for numpy
    """
    import del_msh_numpy.TriMesh

    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.torus(1.0, 0.3, 64, 32)
    tri2vtx = tri2vtx.astype(numpy.uint32)
    vtx2vtx = del_msh_dlpack.Vtx2Vtx.numpy.from_uniform_mesh(
        tri2vtx, vtx2xyz.shape[0], False
    )
    assert vtx2vtx[0].dtype == vtx2vtx[1].dtype == numpy.uint32
    #
    vtx2rhs = numpy.random.rand(vtx2xyz.shape[0], 3).astype(numpy.float32)
    vtx2lhs = numpy.zeros_like(vtx2rhs)
    vtx2lhs_tmp = vtx2lhs.copy()
    #
    lambda0 = 1.0
    del_msh_dlpack.Vtx2Vtx.numpy.laplacian_smoothing(
        *vtx2vtx, lambda0, vtx2lhs, vtx2rhs, 100, vtx2lhs_tmp
    )
    #
    residual = numpy.linalg.norm(
        vtx2lhs
        + lambda0
        * del_msh_dlpack.Vtx2Vtx.numpy.multiply_graph_laplacian(*vtx2vtx, vtx2lhs)
        - vtx2rhs
    )
    assert residual < 3.0e-5


def test_02():
    """
    test Vtx2Vtx for pytorch
    """
    import del_msh_numpy.TriMesh
    import del_msh_dlpack.Vtx2Vtx.torch

    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.torus(1.0, 0.3, 64, 32)
    tri2vtx = torch.from_numpy(tri2vtx).to(torch.uint32)
    vtx2xyz = torch.from_numpy(vtx2xyz)
    vtx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(
        tri2vtx, vtx2xyz.shape[0], False
    )
    assert vtx2vtx[0].dtype == vtx2vtx[1].dtype == torch.uint32
    #
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3)).to(torch.float32)
    vtx2lhs = torch.zeros_like(vtx2rhs)
    vtx2lhstmp = vtx2lhs.clone()
    #
    lambda0 = 1.0
    # [I + lambda * L] {vtx2lhs} = {vtx2rhs}
    # where L = [ .., -1, .., valence, ..,-1, .. ]
    del_msh_dlpack.Vtx2Vtx.torch.laplacian_smoothing(
        *vtx2vtx, lambda0, vtx2lhs, vtx2rhs, 100, vtx2lhstmp
    )
    l_vtx2lhs = del_msh_dlpack.Vtx2Vtx.torch.multiply_graph_laplacian(*vtx2vtx, vtx2lhs)
    res = vtx2lhs + lambda0 * l_vtx2lhs - vtx2rhs
    assert torch.norm(res) < 3.0e-5
    #
    if torch.cuda.is_available():
        print("test laplacian smoothing on gpu")
        vtx2vtx = (vtx2vtx[0].cuda(), vtx2vtx[1].cuda())
        vtx2rhs = vtx2rhs.cuda()
        vtx2lhs = torch.zeros_like(vtx2rhs).cuda()
        # [I + lambda * L] {vtx2lhs} = {vtx2rhs}
        # where L = [ .., -1, .., valence, ..,-1, .. ]
        del_msh_dlpack.Vtx2Vtx.torch.laplacian_smoothing(
            *vtx2vtx, lambda0, vtx2lhs, vtx2rhs, 100, None
        )
        l_vtx2lhs = del_msh_dlpack.Vtx2Vtx.torch.multiply_graph_laplacian(*vtx2vtx, vtx2lhs)
        res = vtx2lhs + lambda0 * l_vtx2lhs - vtx2rhs
        assert torch.norm(res) < 3.0e-5


def test_03():
    """
    test Vtx2Vtx for pytorch
    """
    import del_msh_numpy.TriMesh
    import del_msh_dlpack.Vtx2Vtx.torch

    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.torus(1.0, 0.3, 3, 3)
    tri2vtx = torch.from_numpy(tri2vtx).to(torch.uint32)
    vtx2xyz = torch.from_numpy(vtx2xyz)
    #
    h_vtx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(
        tri2vtx, vtx2xyz.shape[0], False
    )
    if torch.cuda.is_available():
        d_vtx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(
            tri2vtx.cuda(), vtx2xyz.shape[0], False
        )
        assert torch.equal(h_vtx2vtx[0], d_vtx2vtx[0].cpu())
        assert torch.equal(h_vtx2vtx[1], d_vtx2vtx[1].cpu())
    #
    h_vtx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(
        tri2vtx, vtx2xyz.shape[0], True
    )
    if torch.cuda.is_available():
        d_vtx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(
            tri2vtx.cuda(), vtx2xyz.shape[0], True
        )
        assert torch.equal(h_vtx2vtx[0], d_vtx2vtx[0].cpu())
        assert torch.equal(h_vtx2vtx[1], d_vtx2vtx[1].cpu())
