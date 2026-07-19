def elem2volume(elem2idx_offset, idx2vtx, vtx2xyz, elem2volume, stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_elem2volume

    polyhedron_mesh_elem2volume(
        elem2idx_offset, idx2vtx, vtx2xyz, elem2volume, stream_ptr
    )


def elem2center(elem2idx_offset, idx2vtx, vtx2xyz, elem2center, stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_elem2center

    polyhedron_mesh_elem2center(
        elem2idx_offset, idx2vtx, vtx2xyz, elem2center, stream_ptr
    )


def bvhnode2aabb_from_bvhnodes(
    elem2idx_offset, idx2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, stream_ptr=0
):
    #
    from ..del_msh_dlpack import polyhedron_mesh_bvhnode2aabb_from_bvhnodes

    polyhedron_mesh_bvhnode2aabb_from_bvhnodes(
        elem2idx_offset, idx2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, stream_ptr
    )


def search_elem_contain_points(
    bvhnodes,
    bvhnode2aabb,
    elem2idx_offset,
    idx2vtx,
    vtx2xyz,
    wtx2xyz,
    wtx2elem,
    wtx2param,
    stream_ptr=0,
):
    #
    from ..del_msh_dlpack import polyhedron_mesh_search_elem_contain_points

    polyhedron_mesh_search_elem_contain_points(
        bvhnodes,
        bvhnode2aabb,
        elem2idx_offset,
        idx2vtx,
        vtx2xyz,
        wtx2xyz,
        wtx2elem,
        wtx2param,
        stream_ptr,
    )


def subdivide(elem2idx_offset, idx2vtx, vtx2xyz, stream_ptr=0):
    #
    from ..del_msh_dlpack import polyhedron_mesh_subdivide

    return polyhedron_mesh_subdivide(elem2idx_offset, idx2vtx, vtx2xyz, stream_ptr)


def interpolate_values_at_points(
    elem2idx_offset, idx2vtx, vtx2value, wtx2elem, wtx2param, wtx2value, stream_ptr=0
):
    #
    from ..del_msh_dlpack import polyhedron_mesh_interpolate_values_at_points

    polyhedron_mesh_interpolate_values_at_points(
        elem2idx_offset, idx2vtx, vtx2value, wtx2elem, wtx2param, wtx2value, stream_ptr
    )
