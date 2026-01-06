def elem2volume(
    elem2idx_offset,
    idx2vtx,
    vtx2xyz,
    elem2volume,
    stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_elem2volume

    polyhedron_mesh_elem2volume(elem2idx_offset, idx2vtx, vtx2xyz, elem2volume, stream_ptr)