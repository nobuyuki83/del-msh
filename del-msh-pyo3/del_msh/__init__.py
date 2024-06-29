import numpy

def centerize_points(vtxxyz2xyz):
    vtxxyz2xyz[:] -= (vtxxyz2xyz.max(axis=0) + vtxxyz2xyz.min(axis=0))*0.5

def centerize_scale_points(vtxxyz2xyz, scale = 1.0):
    vtxxyz2xyz[:] -= (vtxxyz2xyz.max(axis=0) + vtxxyz2xyz.min(axis=0))*0.5
    vtxxyz2xyz *= scale/(vtxxyz2xyz.max(axis=0) - vtxxyz2xyz.min(axis=0)).max()
    return vtxxyz2xyz

def fit_into_unit_cube(tri2center):
    """
    fit the points inside unit cube [0,1]^3
    :param tri2center:
    :return:
    """
    vmin = numpy.min(tri2center, axis=0)
    vmax = numpy.max(tri2center, axis=0)
    tri2center -= (vmin+vmax)*0.5
    tri2center *= 0.99/(vmax - vmin).max()
    tri2center += numpy.array([0.5, 0.5, 0.5])

def extract_submesh(tri2vtx, tri2bool, vtx2xyz):
    assert tri2vtx.shape[0] == tri2bool.shape[0]
    tri_new2vtx_old = tri2vtx[tri2bool]
    vtx_new2vtx_old = list(set((tri_new2vtx_old[:][:]).flatten()))
    num_vtx_new = len(vtx_new2vtx_old)
    vtx_new2xyz = numpy.zeros((num_vtx_new, 3), dtype=numpy.float32)
    vtx_new2xyz[:][:] = vtx2xyz[vtx_new2vtx_old[:]][:]
    vtx_old2vtx_new = numpy.full(shape=vtx2xyz.shape[0],
                                 fill_value=numpy.iinfo(numpy.uint64).max,
                                 dtype=numpy.uint64)
    for vtx_new, vtx_old in enumerate(vtx_new2vtx_old):
        vtx_old2vtx_new[vtx_old] = vtx_new
    tri_new2vtx_new = numpy.zeros_like(tri_new2vtx_old)
    tri_new2vtx_new[:][:] = vtx_old2vtx_new[tri_new2vtx_old[:][:]]
    return tri_new2vtx_new, vtx_new2xyz, vtx_new2vtx_old

