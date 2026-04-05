


def from_vtx2vtx(idx2vtx_offset, vtx2idx, edge2vtx, stream_ptr=0):
    from ..del_msh_dlpack import edge2vtx_from_vtx2vtx

    return edge2vtx_from_vtx2vtx(idx2vtx_offset, vtx2idx, edge2vtx, stream_ptr)


def contour_for_triangle_mesh(tri2vtx, vtx2xyz, transform_world2ndc, edge2vtx, edge2tri, stream_ptr=0):
    from ..del_msh_dlpack import edge2vtx_contour_for_triangle_mesh

    return edge2vtx_contour_for_triangle_mesh(tri2vtx, vtx2xyz, transform_world2ndc, edge2vtx, edge2tri, stream_ptr)