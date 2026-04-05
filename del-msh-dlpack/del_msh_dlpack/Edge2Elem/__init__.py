



def from_edge2vtx_of_tri2vtx_with_vtx2vtx(edge2vtx, tri2vtx, vtx2idx_offset, idx2vtx, edge2tri, stream_ptr=0):
    from ..del_msh_dlpack import edge2elem_from_edge2vtx_of_tri2vtx_with_vtx2vtx

    edge2elem_from_edge2vtx_of_tri2vtx_with_vtx2vtx(
        edge2vtx,
        tri2vtx,
        vtx2idx_offset,
        idx2vtx,
        edge2tri,
        stream_ptr
    )
