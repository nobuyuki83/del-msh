import numpy as np
from .. import _CapsuleAsDLPack

def from_uniform_mesh(elem2vtx: np.ndarray, num_vtx: int, is_self: bool):
    """make vertex surrounding vertex data from uniform mesh
    """
    assert len(elem2vtx.shape) == 2
    assert elem2vtx.dtype == np.int32
    from .. import Vtx2Vtx
    cap_vtx2idx, cap_idx2vtx = Vtx2Vtx.from_uniform_mesh(
        elem2vtx.__dlpack__(),
        num_vtx,
        is_self
    )
    vtx2idx = np.from_dlpack(_CapsuleAsDLPack(cap_vtx2idx)).copy()
    idx2vtx = np.from_dlpack(_CapsuleAsDLPack(cap_idx2vtx)).copy()
    return vtx2idx, idx2vtx

def laplacian_smoothing(
    vtx2idx: np.ndarray,
    idx2vtx: np.ndarray,
    lambda0: float,
    vtx2lhs: np.ndarray,
    vtx2rhs: np.ndarray,
    num_iter: int,
    vtx2lhstmp: np.ndarray | None):
    """Solve the linear system from screened Poisson equation using Jacobi method:
    [I + lambda * L] {vtx2lhs} = {vtx2rhs}
    where L = [ .., -1, .., valence, ..,-1, .. ]
    """
    num_vtx = vtx2idx.shape[0] - 1
    #
    assert len(vtx2idx.shape) == 1
    assert len(idx2vtx.shape) == 1
    assert vtx2lhs.shape == vtx2rhs.shape
    assert len(vtx2lhs.shape) == 2
    assert vtx2lhs.shape[0] == num_vtx
    assert vtx2lhs.shape == vtx2rhs.shape
    assert vtx2idx.dtype == np.int32
    assert idx2vtx.dtype == np.int32
    assert vtx2lhs.dtype == vtx2rhs.dtype == np.float32
    assert num_iter >= 0
    if vtx2lhstmp is None:
        vtx2lhstmp = np.zeros_like(vtx2lhs)
    from .. import Vtx2Vtx
    Vtx2Vtx.laplacian_smoothing(
        vtx2idx.__dlpack__(),
        idx2vtx.__dlpack__(),
        lambda0,
        vtx2lhs.__dlpack__(),
        vtx2rhs.__dlpack__(),
        num_iter,
        vtx2lhstmp.__dlpack__())


def multiply_graph_laplacian(
    vtx2idx: np.ndarray,
    idx2vtx: np.ndarray,
    vtx2rhs: np.ndarray) -> np.ndarray:
    num_vtx = vtx2idx.shape[0] - 1
    assert len(vtx2idx.shape) == 1
    assert len(idx2vtx.shape) == 1
    assert len(vtx2rhs.shape) == 2
    assert vtx2rhs.shape[0] == num_vtx
    assert vtx2idx.dtype == np.int32
    assert idx2vtx.dtype == np.int32
    assert vtx2rhs.dtype == np.float32
    #
    vtx2lhs = np.zeros_like(vtx2rhs)
    #
    from .. import Vtx2Vtx
    Vtx2Vtx.multiply_graph_laplacian(
        vtx2idx.__dlpack__(),
        idx2vtx.__dlpack__(),
        vtx2rhs.__dlpack__(),
        vtx2lhs.__dlpack__())
    return vtx2lhs