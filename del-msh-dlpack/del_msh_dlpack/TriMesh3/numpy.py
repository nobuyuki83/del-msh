import numpy as np


def tri2normal(tri2vtx: np.ndarray, vtx2xyz: np.ndarray):
    num_tri = tri2vtx.shape[0]
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    assert tri2vtx.dtype == np.uint32
    assert len(vtx2xyz.shape) == 2
    assert vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == np.float32
    tri2nrm = np.zeros(shape=(num_tri, 3), dtype=np.float32)
    from .. import TriMesh3

    TriMesh3.tri2normal(
        tri2vtx.__dlpack__(), vtx2xyz.__dlpack__(), tri2nrm.__dlpack__()
    )
    return tri2nrm


def bwd_tri2normal(tri2vtx: np.ndarray, vtx2xyz: np.ndarray, dw_tri2nrm: np.ndarray):
    # num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    assert tri2vtx.dtype == np.uint32
    assert len(vtx2xyz.shape) == 2
    assert vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == np.float32
    assert len(dw_tri2nrm.shape) == 2
    assert dw_tri2nrm.shape[1] == 3
    assert dw_tri2nrm.dtype == np.float32
    dw_vtx2xyz = np.ndarray(shape=(num_vtx, 3), dtype=np.float32)
    from .. import TriMesh3

    TriMesh3.bwd_tri2normal(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        dw_tri2nrm.__dlpack__(),
        dw_vtx2xyz.__dlpack__(),
    )
    return dw_vtx2xyz
