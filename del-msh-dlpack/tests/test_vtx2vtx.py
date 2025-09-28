import numpy
import torch
#

def test_01():
    '''
    test Vtx2Vtx for numpy
    '''
    import del_msh_numpy.TriMesh
    import del_msh_dlpack.Vtx2Vtx.np
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.torus(1.0, 0.3, 64, 32)
    vtx2vtxA = del_msh_dlpack.Vtx2Vtx.np.from_uniform_mesh(tri2vtx.astype(numpy.uint64), vtx2xyz.shape[0], False)
    assert vtx2vtxA[0].dtype == vtx2vtxA[1].dtype == numpy.uint64
    vtx2vtxB = del_msh_dlpack.Vtx2Vtx.np.from_uniform_mesh(tri2vtx.astype(numpy.uint32), vtx2xyz.shape[0], False)
    assert vtx2vtxB[0].dtype == vtx2vtxB[1].dtype == numpy.uint32
    assert numpy.array_equal(vtx2vtxA[0], vtx2vtxB[0].astype(numpy.uint64) )
    assert numpy.array_equal(vtx2vtxA[1], vtx2vtxB[1].astype(numpy.uint64) )
    vtx2vtx = vtx2vtxB
    #
    vtx2rhs = numpy.random.rand(vtx2xyz.shape[0], 3).astype(numpy.float32)
    vtx2lhs = numpy.zeros_like(vtx2rhs)
    vtx2lhs_tmp = vtx2lhs.copy()
    #
    lambda0 = 1.0
    del_msh_dlpack.Vtx2Vtx.np.laplacian_smoothing(
        *vtx2vtx, lambda0, vtx2lhs, vtx2rhs, 100, vtx2lhs_tmp)
    #
    residual = numpy.linalg.norm(\
    vtx2lhs + lambda0 * del_msh_dlpack.Vtx2Vtx.np.multiply_graph_laplacian(*vtx2vtx, vtx2lhs)\
     - vtx2rhs)
    assert residual < 3.0e-5


def test_02():
    '''
    test Vtx2Vtx for pytorch
    '''
    import del_msh_numpy.TriMesh
    import del_msh_dlpack.Vtx2Vtx.pt
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.torus(1.0, 0.3, 64, 32)
    # print(vtx2xyz.shape)
    tri2vtx = torch.from_numpy(tri2vtx)
    vtx2xyz = torch.from_numpy(vtx2xyz)
    vtx2vtxA = del_msh_dlpack.Vtx2Vtx.pt.from_uniform_mesh(tri2vtx.to(torch.uint64), vtx2xyz.shape[0], False)
    assert vtx2vtxA[0].dtype == vtx2vtxA[1].dtype == torch.uint64
    vtx2vtxB = del_msh_dlpack.Vtx2Vtx.pt.from_uniform_mesh(tri2vtx.to(torch.uint32), vtx2xyz.shape[0], False)
    assert vtx2vtxB[0].dtype == vtx2vtxB[1].dtype == torch.uint32
    vtx2vtx = vtx2vtxB
    #
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3)).to(torch.float32)
    vtx2lhs = torch.zeros_like(vtx2rhs)
    vtx2lhstmp = vtx2lhs.clone()
    #
    lambda0 = 1.0
    del_msh_dlpack.Vtx2Vtx.pt.laplacian_smoothing(
        *vtx2vtx, lambda0, vtx2lhs, vtx2rhs, 100, vtx2lhstmp)
    #print(vtx2rhs)
    #
    residual = numpy.linalg.norm(\
    vtx2lhs + lambda0 * del_msh_dlpack.Vtx2Vtx.np.multiply_graph_laplacian(*vtx2vtx, vtx2lhs)\
     - vtx2rhs)
    assert residual < 3.0e-5
    #
    if torch.cuda.is_available():
        print("test laplacian smoothing on gpu")
        vtx2vtx = (vtx2vtx[0].cuda(), vtx2vtx[1].cuda())
        vtx2rhs = vtx2rhs.cuda()
        vtx2lhs = torch.zeros_like(vtx2rhs).cuda()
        del_msh_dlpack.Vtx2Vtx.pt.laplacian_smoothing(
            *vtx2vtx, lambda0, vtx2lhs, vtx2rhs, 100, None)
        vtx2lhs = vtx2lhs.cpu().numpy()
        vtx2vtx = (vtx2vtx[0].cpu().numpy(), vtx2vtx[1].cpu().numpy())
        vtx2rhs = vtx2rhs.cpu().numpy()
        residual = numpy.linalg.norm(\
        vtx2lhs + lambda0 * del_msh_dlpack.Vtx2Vtx.np.multiply_graph_laplacian(*vtx2vtx, vtx2lhs)\
         - vtx2rhs)
        # print(residual)
        assert residual < 3.0e-5