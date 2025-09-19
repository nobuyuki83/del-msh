import numpy
from . import _CapsuleAsDLPack

def laplacian_smoothing(
    vtx2idx,
    idx2vtx,
    lambda0: float,
    vtx2lhs,
    vtx2rhs,
    num_iter: int,
    vtx2lhs_tmp):
    #
    from .del_msh_dlpack import vtx2vtx_laplacian_smoothing
    vtx2vtx_laplacian_smoothing(
        vtx2idx.__dlpack__(),
        idx2vtx.__dlpack__(),
        lambda0,
        vtx2lhs.__dlpack__(),
        vtx2rhs.__dlpack__(),
        num_iter,
        vtx2lhs_tmp.__dlpack__())

def from_uniform_mesh(elem2vtx, num_vtx, is_self):
    from .del_msh_dlpack import vtx2vtx_from_uniform_mesh
    cap_vtx2idx, cap_idx2vtx = vtx2vtx_from_uniform_mesh(
        elem2vtx.__dlpack__(),
        num_vtx,
        is_self
    )
    vtx2idx = numpy.from_dlpack(_CapsuleAsDLPack(cap_vtx2idx)).copy()
    idx2vtx = numpy.from_dlpack(_CapsuleAsDLPack(cap_idx2vtx)).copy()
    return vtx2idx, idx2vtx