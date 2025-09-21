import numpy
#
import del_msh_numpy.TriMesh
import del_msh_dlpack.Vtx2Vtx.np

def test_01():
    '''
    test raycast
    '''
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.torus(1.0, 0.3, 64, 32)
    vtx2vtxA = del_msh_dlpack.Vtx2Vtx.np.from_uniform_mesh(tri2vtx.astype(numpy.uint64), vtx2xyz.shape[0], False)
    assert vtx2vtxA[0].dtype == vtx2vtxA[1].dtype == numpy.uint64
    vtx2vtxB = del_msh_dlpack.Vtx2Vtx.np.from_uniform_mesh(tri2vtx.astype(numpy.uint32), vtx2xyz.shape[0], False)
    assert vtx2vtxB[0].dtype == vtx2vtxB[1].dtype == numpy.uint32
    assert numpy.array_equal(vtx2vtxA[0], vtx2vtxB[0].astype(numpy.uint64) )
    assert numpy.array_equal(vtx2vtxA[1], vtx2vtxB[1].astype(numpy.uint64) )
    vtx2vtx = vtx2vtxB
    #
    vtx2xyz = vtx2xyz.astype(numpy.float32)
    vtx2rhs = numpy.random.rand(vtx2xyz.shape[0], 3).astype(numpy.float32)
    vtx2lhs = numpy.zeros_like(vtx2rhs)
    vtx2lhs_tmp = vtx2lhs.copy()
    #
    lambda0 = 1.0
    del_msh_dlpack.Vtx2Vtx.np.laplacian_smoothing(
        vtx2vtx[0], vtx2vtx[1], lambda0, vtx2lhs, vtx2rhs, 100, vtx2lhs_tmp)
    #
    residual = numpy.linalg.norm(\
    vtx2lhs + lambda0 * del_msh_dlpack.Vtx2Vtx.np.multiply_graph_laplacian(vtx2vtx[0], vtx2vtx[1], vtx2lhs)\
     - vtx2rhs)
    #
    assert residual < 3.0e-5