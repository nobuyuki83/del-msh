def vtx2morton_from_vtx2co(vtx2co, transform_co2unit, vtx2morton, stream_ptr=0):
    from ..del_msh_dlpack import mortons_vtx2morton_from_vtx2co

    vtx2morton = mortons_vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit, vtx2morton, stream_ptr
    )
    return vtx2morton


def make_bvh(idx2obj, idx2morton, bvhnodes, stream_ptr=0):
    from ..del_msh_dlpack import mortons_make_bvh

    mortons_make_bvh(idx2obj, idx2morton, bvhnodes, stream_ptr)
