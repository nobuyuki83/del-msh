import numpy as np
from .. import util_numpy

def tri2centroid(tri2vtx: np.ndarray, vtx2xyz: np.ndarray) -> np.ndarray:
    """compute centroids of triangles

    Args:
        tri2vtx: (num_tri, 3) uint32
        vtx2xyz: (num_vtx, 3) float32
    Returns:
        tri2centroid: (num_tri, 3) float32
    """
    assert tri2vtx.shape[1] == 3 and tri2vtx.dtype == np.uint32
    assert vtx2xyz.shape[1] == 3 and vtx2xyz.dtype == np.float32
    idx = tri2vtx.astype(np.int64)
    return (vtx2xyz[idx[:, 0]] + vtx2xyz[idx[:, 1]] + vtx2xyz[idx[:, 2]]) / 3.0


def tri2normal(tri2vtx: np.ndarray, vtx2xyz: np.ndarray):
    num_tri = tri2vtx.shape[0]
    #
    assert tri2vtx.shape == (num_tri, 3) and tri2vtx.dtype == np.uint32
    assert len(vtx2xyz.shape) == 2
    assert vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == np.float32
    #
    tri2nrm = np.zeros(shape=(num_tri, 3), dtype=np.float32)
    #
    from .. import TriMesh3

    TriMesh3.tri2normal(
        tri2vtx.__dlpack__(), vtx2xyz.__dlpack__(), tri2nrm.__dlpack__()
    )
    return tri2nrm


def bwd_tri2normal(tri2vtx: np.ndarray, vtx2xyz: np.ndarray, dw_tri2nrm: np.ndarray):
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    #
    assert tri2vtx.shape == (num_tri, 3) and tri2vtx.dtype == np.uint32
    assert vtx2xyz.shape == (num_vtx, 3) and vtx2xyz.dtype == np.float32
    assert len(dw_tri2nrm.shape) == 2
    assert dw_tri2nrm.shape[1] == 3
    assert dw_tri2nrm.dtype == np.float32
    #
    dw_vtx2xyz = np.ndarray(shape=(num_vtx, 3), dtype=np.float32)
    #
    from .. import TriMesh3

    TriMesh3.bwd_tri2normal(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        dw_tri2nrm.__dlpack__(),
        dw_vtx2xyz.__dlpack__(),
    )
    return dw_vtx2xyz


def bvh_aabb(tri2vtx: np.ndarray, vtx2xyz: np.ndarray):
    num_vtx = vtx2xyz.shape[0]
    num_tri = tri2vtx.shape[0]
    util_numpy.assert_shape_dtype(tri2vtx, (num_tri,3), np.uint32)
    util_numpy.assert_shape_dtype(vtx2xyz, (num_vtx,3), np.float32)
    #
    tri2cog = tri2centroid(tri2vtx, vtx2xyz)
    print(tri2cog.shape)
    from ..Vtx2Xyz.numpy import normalize_to_unit_cube
    transform_co2unit = normalize_to_unit_cube(vtx2xyz)
    #
    from .. import Mortons
    from ..Mortons import numpy
    tri2morton = Mortons.numpy.vtx2morton_from_vtx2co(tri2cog, transform_co2unit)
    idx2tri = np.argsort(tri2morton)
    idx2morton = tri2morton[idx2tri]
    idx2tri = idx2tri.astype(np.uint32)
    bvhnodes = Mortons.numpy.make_bvh(idx2tri, idx2morton)
    num_bvhnodes = bvhnodes.shape[0]
    #
    bvhnode2aabb = np.ndarray(shape=(num_bvhnodes, 6), dtype=np.float32)
    vtx2xyz1 = np.zeros(shape=(0,3), dtype=np.float32)
    from .. import TriMesh3
    TriMesh3.aabb_from_bvhnodes(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        vtx2xyz1.__dlpack__(),
        bvhnodes.__dlpack__(),
        bvhnode2aabb.__dlpack__(),
    )
    return bvhnodes, bvhnode2aabb