import numpy

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
