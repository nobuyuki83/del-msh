def update_pix2tri(
    tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, pix2tri, stream_ptr=0
):
    from ..del_msh_dlpack import pix2tri_by_raycast

    pix2tri_by_raycast(
        pix2tri,
        tri2vtx,
        vtx2xyz,
        bvhnodes,
        bvhnode2aabb,
        transform_ndc2world,
        stream_ptr,
    )


def interpolate(
    pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world, pix2val, stream_ptr=0
):
    from ..del_msh_dlpack import pix2tri_interpolate

    pix2tri_interpolate(
        pix2tri, tri2vtx, vtx2xyz, vtx2val, transform_ndc2world, pix2val, stream_ptr
    )


def interpolate_bwd(
    pix2tri,
    tri2vtx,
    vtx2xyz,
    vtx2val,
    transform_ndc2world,
    dldw_pix2val,
    dldw_vtx2xyz,
    dldw_vtx2val,
    stream_ptr=0,
):
    from ..del_msh_dlpack import pix2tri_interpolate_bwd

    pix2tri_interpolate_bwd(
        pix2tri,
        tri2vtx,
        vtx2xyz,
        vtx2val,
        transform_ndc2world,
        dldw_pix2val,
        dldw_vtx2xyz,
        dldw_vtx2val,
        stream_ptr,
    )
