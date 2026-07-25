import numpy as np
import numpy.typing as npt
from .. import _CapsuleAsDLPack
from ..util_numpy import assert_shape_dtype


def from_uniform_mesh(elem2vtx: npt.NDArray, num_vtx: int, is_self: bool):
    """make vertex surrounding vertex data from uniform mesh"""
    assert len(elem2vtx.shape) == 2
    assert elem2vtx.dtype == np.uint32
    from .. import Vtx2Vtx

    cap_vtx2idx, cap_idx2vtx = Vtx2Vtx.from_uniform_mesh(
        elem2vtx.__dlpack__(), num_vtx, is_self
    )
    vtx2idx = np.from_dlpack(_CapsuleAsDLPack(cap_vtx2idx)).copy()
    idx2vtx = np.from_dlpack(_CapsuleAsDLPack(cap_idx2vtx)).copy()
    return vtx2idx, idx2vtx


def graph_screened_poisson(
    vtx2idx: npt.NDArray,
    idx2vtx: npt.NDArray,
    lambda0: float,
    vtx2lhs: npt.NDArray,
    vtx2rhs: npt.NDArray,
    num_iter: int,
    vtx2lhstmp: npt.NDArray | None,
):
    """Solve the linear system from screened Poisson equation using Jacobi method:
    [I + lambda * L] {vtx2lhs} = {vtx2rhs}
    where L = [ .., -1, .., valence, ..,-1, .. ]
    """
    num_vtx = vtx2idx.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_vdim = vtx2lhs.shape[1]
    #
    assert_shape_dtype(vtx2idx, (num_vtx + 1,), np.uint32)
    assert_shape_dtype(idx2vtx, (num_idx,), np.uint32)
    assert_shape_dtype(vtx2lhs, (num_vtx, num_vdim), np.float32)
    assert_shape_dtype(vtx2rhs, (num_vtx, num_vdim), np.float32)

    if vtx2lhstmp is None:
        vtx2lhstmp = np.zeros_like(vtx2lhs)
    else:
        assert_shape_dtype(vtx2lhstmp, (num_vtx, num_vdim), np.float32)
    #
    from .. import Vtx2Vtx

    Vtx2Vtx.graph_screened_poisson(
        vtx2idx.__dlpack__(),
        idx2vtx.__dlpack__(),
        lambda0,
        vtx2lhs.__dlpack__(),
        vtx2rhs.__dlpack__(),
        num_iter,
        vtx2lhstmp.__dlpack__(),
    )


def multiply_graph_laplacian(
    vtx2idx: npt.NDArray, idx2vtx: npt.NDArray, vtx2rhs: npt.NDArray
) -> np.ndarray:
    num_vtx = vtx2idx.shape[0] - 1
    assert len(vtx2idx.shape) == 1
    assert len(idx2vtx.shape) == 1
    assert len(vtx2rhs.shape) == 2
    assert vtx2rhs.shape[0] == num_vtx
    assert vtx2idx.dtype == np.uint32
    assert idx2vtx.dtype == np.uint32
    assert vtx2rhs.dtype == np.float32
    #
    vtx2lhs = np.zeros_like(vtx2rhs)
    #
    from .. import Vtx2Vtx

    Vtx2Vtx.multiply_graph_laplacian(
        vtx2idx.__dlpack__(),
        idx2vtx.__dlpack__(),
        vtx2rhs.__dlpack__(),
        vtx2lhs.__dlpack__(),
    )
    return vtx2lhs
