import numpy


def laplacian_smoothing(
    vtx2idx,
    idx2vtx,
    lambda0: float,
    vtx2lhs,
    vtx2rhs,
    num_iter: int,
    vtx2lhs_tmp,
    stream_ptr=0,
):
    """Solve the linear system from screened Poisson equation using Jacobi method:
    [I + lambda * L] {vtx2lhs} = {vtx2rhs}
    where L = [ .., -1, .., valence, ..,-1, .. ]
    """
    from ..del_msh_dlpack import vtx2vtx_laplacian_smoothing

    vtx2vtx_laplacian_smoothing(
        vtx2idx, idx2vtx, lambda0, vtx2lhs, vtx2rhs, num_iter, vtx2lhs_tmp, stream_ptr
    )


def multiply_graph_laplacian(vtx2idx, idx2vtx, vtx2rhs, vtx2lhs):
    """Multiply graph laplacian matrix to vector
    {vtx2lhs} = L * {vtx2rhs}
    where L = [ .., -1, .., valence, ..,-1, .. ]
    """
    from ..del_msh_dlpack import vtx2vtx_multiply_graph_laplacian

    vtx2vtx_multiply_graph_laplacian(vtx2idx, idx2vtx, vtx2rhs, vtx2lhs)


def from_uniform_mesh(elem2vtx, num_vtx: int, is_self, stream_ptr=0):
    from ..del_msh_dlpack import vtx2vtx_from_uniform_mesh

    cap_vtx2idx, cap_idx2vtx = vtx2vtx_from_uniform_mesh(
        elem2vtx, num_vtx, is_self, stream_ptr
    )
    return cap_vtx2idx, cap_idx2vtx
