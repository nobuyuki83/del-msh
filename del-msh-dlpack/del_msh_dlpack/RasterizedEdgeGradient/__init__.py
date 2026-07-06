

def bwd(tri2vtx, vtx2xyz, dldw_vtx2xyz, transform_world2pix, dldw_pixval, pix2tri):
    from ..del_msh_dlpack import rasterized_edge_gradient_bwd
    rasterized_edge_gradient_bwd(
        tri2vtx,
        vtx2xyz,
        dldw_vtx2xyz,
        transform_world2pix,
        dldw_pixval,
        pix2tri,
    )


def edge_gradient_and_type(
    tri2vtx, vtx2xyz, transform_world2pix, dldw_pixval, pix2tri,
    hedge2type, hedge2dldr, vedge2type, vedge2dldr,
):
    from ..del_msh_dlpack import rasterized_edge_gradient_edge_gradient_and_type
    rasterized_edge_gradient_edge_gradient_and_type(
        tri2vtx,
        vtx2xyz,
        transform_world2pix,
        dldw_pixval,
        pix2tri,
        hedge2type,
        hedge2dldr,
        vedge2type,
        vedge2dldr,
    )


def smooth_gradient(hedge2type, hedge2dldr, vedge2type, vedge2dldr):
    from ..del_msh_dlpack import rasterized_edge_gradient_smooth_gradient
    rasterized_edge_gradient_smooth_gradient(
        hedge2type,
        hedge2dldr,
        vedge2type,
        vedge2dldr,
    )