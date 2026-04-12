def update(
        pix2depth,
        pix2tri,
        tri2vtx,
        vtx2xyz,
        transform_ndc2world,
        stream_ptr=0):
    from ..del_msh_dlpack import pix2depth_update
    pix2depth_update(
        pix2depth,
        pix2tri,
        tri2vtx,
        vtx2xyz,
        transform_ndc2world,
        stream_ptr
    )


def bwd_wrt_vtx2xyz(
        dldw_vtx2xyz,
        pix2tri,
        tri2vtx,
        vtx2xyz,
        dldw_pix2depth,
        transform_ndc2world,
        stream_ptr=0):
    from ..del_msh_dlpack import pix2depth_bwd_wrt_vtx2xyz
    pix2depth_bwd_wrt_vtx2xyz(
        dldw_vtx2xyz,
        pix2tri,
        tri2vtx,
        vtx2xyz,
        dldw_pix2depth,
        transform_ndc2world,
        stream_ptr
    )
