import numpy
#
import del_msh_numpy
import del_msh_dlpack.Vtx2Vtx

def test_01():
    '''
    test raycast
    '''
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.torus(1.0, 0.3, 64, 32)
    vtx2xyz = vtx2xyz.astype(numpy.float32)
    print(vtx2xyz.shape)
    vtx2vtx = del_msh_numpy.TriMesh.vtx2vtx(tri2vtx, vtx2xyz.shape[0], False)
    print(vtx2vtx[0].dtype)
    vtx2rhs = numpy.random.rand(vtx2xyz.shape[0], 3).astype(numpy.float32)
    vtx2lhs = numpy.zeros_like(vtx2rhs)
    vtx2lhs_tmp = vtx2lhs.copy()
    #
    print(vtx2lhs)
    print(vtx2rhs)
    print(vtx2lhs.ctypes.data)
    del_msh_dlpack.Vtx2Vtx.laplacian_smoothing(
        vtx2vtx[0], vtx2vtx[1], 1.0, vtx2lhs, vtx2rhs, 100, vtx2lhs_tmp)
    print(vtx2lhs.ctypes.data)
    print(vtx2lhs)