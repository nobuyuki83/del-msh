
def antialias(edge2vtx_contour, vtx2xyz, transform_world2pix, pix2tri, img_data, stream_ptr=0):
    from ..del_msh_dlpack import differential_rasterizer_antialias

    differential_rasterizer_antialias(edge2vtx_contour, vtx2xyz, transform_world2pix, pix2tri, img_data, stream_ptr)


def bwd_antialias(edge2vtx_contour, vtx2xyz, dldw_vtx2xyz, transform_world2pix, dldw_pixval, pix2tri):
    from ..del_msh_dlpack import differential_rasterizer_bwd_antialias

    differential_rasterizer_bwd_antialias(edge2vtx_contour, vtx2xyz, dldw_vtx2xyz, transform_world2pix, dldw_pixval, pix2tri)
