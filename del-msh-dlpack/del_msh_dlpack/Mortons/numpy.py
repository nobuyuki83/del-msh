import numpy as np
from .. import util_numpy


def vtx2morton_from_vtx2co(vtx2co: np.ndarray, transform_co2unit: np.ndarray) -> np.ndarray:
    num_vtx = vtx2co.shape[0]
    num_dim = vtx2co.shape[1]
    #
    assert num_dim == 2 or num_dim == 3
    util_numpy.assert_shape_dtype(vtx2co, shape=(num_vtx, num_dim), dtype=np.float32)
    util_numpy.assert_shape_dtype(transform_co2unit, shape=(num_dim+1, num_dim+1), dtype=np.float32)
    #
    vtx2morton = np.empty((num_vtx,), dtype=np.uint32)
    from .. import Mortons

    Mortons.vtx2morton_from_vtx2co(
        vtx2co.__dlpack__(),
        np.ascontiguousarray(transform_co2unit.T).__dlpack__(),
        vtx2morton.__dlpack__(),
    )
    return vtx2morton


def make_bvh(idx2obj: np.ndarray, idx2morton: np.ndarray) -> np.ndarray:
    n = idx2obj.shape[0]
    #
    assert idx2obj.dtype == np.uint32 and idx2obj.ndim == 1
    assert idx2morton.dtype == np.uint32 and idx2morton.ndim == 1
    assert idx2obj.shape == idx2morton.shape
    #
    bvhnodes = np.empty((n * 2 - 1, 3), dtype=np.uint32)
    from .. import Mortons

    Mortons.make_bvh(
        idx2obj.__dlpack__(),
        idx2morton.__dlpack__(),
        bvhnodes.__dlpack__(),
    )
    return bvhnodes
