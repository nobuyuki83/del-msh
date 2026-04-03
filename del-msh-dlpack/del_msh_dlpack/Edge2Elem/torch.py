


def from_edge2vtx_of_tri2vtx_with_vtx2vtx(
        edge2vtx,
        tri2vtx,
        vtx2idx_offset,
        idx2vtx,
        edge2tri):
    from .. import Edge2Elem

    Edge2Elem.from_edge2vtx_of_tri2vtx_with_vtx2vtx(
        edge2vtx.__dlpack__(),
        tri2vtx.__dlpack__(),
        vtx2idx_offset.__dlpack__(),
        idx2vtx.__dlpack__(),
        edge2tri.__dlpack__()
    )
