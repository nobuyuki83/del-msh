def bwd(
    tri2vtx,
    vtx2xyz,
    dldw_vtx2xyz,
    transform_world2pix,
    pix2tri,
    pix2val,
    dldw_pix2val,
    stream_ptr=0,
):
    from ..del_msh_dlpack import edgegrad_bwd

    edgegrad_bwd(
        tri2vtx,
        vtx2xyz,
        dldw_vtx2xyz,
        transform_world2pix,
        pix2tri,
        pix2val,
        dldw_pix2val,
        stream_ptr,
    )
