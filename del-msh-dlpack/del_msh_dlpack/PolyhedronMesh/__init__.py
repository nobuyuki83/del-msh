def elem2volume(
    elem2idx_offset,
    idx2vtx,
    vtx2xyz,
    elem2volume,
    stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_elem2volume

    polyhedron_mesh_elem2volume(elem2idx_offset, idx2vtx, vtx2xyz, elem2volume, stream_ptr)


def elem2center(
    elem2idx_offset,
    idx2vtx,
    vtx2xyz,
    elem2center,
    stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_elem2center

    polyhedron_mesh_elem2center(elem2idx_offset, idx2vtx, vtx2xyz, elem2center, stream_ptr)


def bvhnode2aabb_from_bvhnodes(
    elem2idx_offset,
    idx2vtx,
    vtx2xyz,
    bvhnodes,
    bvhnode2aabb,
    stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_bvhnode2aabb_from_bvhnodes

    polyhedron_mesh_bvhnode2aabb_from_bvhnodes(
        elem2idx_offset, idx2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, stream_ptr)


def nearest_elem_for_points(
    bvhnodes,
    bvhnode2aabb,
    elem2idx_offset,
    idx2vtx,
    vtx2xyz,
    wtx2xyz,
    wtx2elem,
    wtx2param,
    stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_nearest_elem_for_points

    polyhedron_mesh_nearest_elem_for_points(
        bvhnodes, bvhnode2aabb, elem2idx_offset, idx2vtx, vtx2xyz,
        wtx2xyz, wtx2elem, wtx2param, stream_ptr)