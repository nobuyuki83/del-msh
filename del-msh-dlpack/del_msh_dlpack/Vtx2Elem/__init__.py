def from_uniform_mesh(elem2vtx, num_vtx: int, stream_ptr=0):
    from ..del_msh_dlpack import vtx2elem_from_uniform_mesh
    cap_vtx2idx, cap_idx2elem = vtx2elem_from_uniform_mesh(
        elem2vtx,
        num_vtx,
        stream_ptr
    )
    return cap_vtx2idx, cap_idx2elem