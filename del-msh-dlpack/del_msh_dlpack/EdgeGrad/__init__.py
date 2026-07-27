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
    from ..del_msh_dlpack import rasterized_edge_gradient_bwd

    rasterized_edge_gradient_bwd(
        tri2vtx,
        vtx2xyz,
        dldw_vtx2xyz,
        transform_world2pix,
        pix2tri,
        pix2val,
        dldw_pix2val,
        stream_ptr,
    )


def edge_gradient_and_type(
    tri2vtx,
    vtx2xyz,
    transform_world2pix,
    pix2tri,
    pix2val,
    dldw_pix2val,
    hedge2type,
    hedge2dldr,
    vedge2type,
    vedge2dldr,
    stream_ptr=0,
):
    from ..del_msh_dlpack import rasterized_edge_gradient_edge_gradient_and_type

    rasterized_edge_gradient_edge_gradient_and_type(
        tri2vtx,
        vtx2xyz,
        transform_world2pix,
        pix2tri,
        pix2val,
        dldw_pix2val,
        hedge2type,
        hedge2dldr,
        vedge2type,
        vedge2dldr,
        stream_ptr,
    )


def smooth_gradient(
    hedge2type, hedge2dldr, vedge2type, vedge2dldr, num_iter, stream_ptr=0
):
    from ..del_msh_dlpack import rasterized_edge_gradient_smooth_gradient

    rasterized_edge_gradient_smooth_gradient(
        hedge2type, hedge2dldr, vedge2type, vedge2dldr, num_iter, stream_ptr
    )


def interpolate(hedge2vy, vedge2vx, vtx2xy, vtx2velo, stream_ptr=0):
    from ..del_msh_dlpack import rasterized_edge_gradient_interpolate

    rasterized_edge_gradient_interpolate(
        hedge2vy, vedge2vx, vtx2xy, vtx2velo, stream_ptr
    )
