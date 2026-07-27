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
    from ..del_msh_dlpack import edgegrad_edge_gradient_and_type

    edgegrad_edge_gradient_and_type(
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
    from ..del_msh_dlpack import edgegrad_smooth_gradient

    edgegrad_smooth_gradient(
        hedge2type, hedge2dldr, vedge2type, vedge2dldr, num_iter, stream_ptr
    )


def interpolate(hedge2vy, vedge2vx, vtx2xy, vtx2velo, stream_ptr=0):
    from ..del_msh_dlpack import edgegrad_interpolate

    edgegrad_interpolate(hedge2vy, vedge2vx, vtx2xy, vtx2velo, stream_ptr)
