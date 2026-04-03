


def from_vtx2vtx(idx2vtx_offset, vtx2idx, edge2vtx):
    from ..del_msh_dlpack import edge2vtx_from_vtx2vtx

    return edge2vtx_from_vtx2vtx(idx2vtx_offset, vtx2idx, edge2vtx)