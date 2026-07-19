def antialias(
    cedge2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin, pix2vout, stream_ptr=0
):
    from ..del_msh_dlpack import differentiable_antialias_fwd

    differentiable_antialias_fwd(
        cedge2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2vin, pix2vout, stream_ptr
    )


def bwd_antialias(
    cedge2vtx,
    vtx2xyz,
    dldw_vtx2xyz,
    transform_world2pix,
    pix2val,
    dldw_pixval,
    pix2tri,
    stream_ptr=0,
):
    from ..del_msh_dlpack import differentiable_antialias_bwd

    differentiable_antialias_bwd(
        cedge2vtx,
        vtx2xyz,
        dldw_vtx2xyz,
        transform_world2pix,
        pix2val,
        dldw_pixval,
        pix2tri,
        stream_ptr,
    )
