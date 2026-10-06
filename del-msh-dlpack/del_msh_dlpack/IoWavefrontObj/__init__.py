def save_trimesh3(tri2vtx, vtx2xyz, path_file):
    from ..del_msh_dlpack import io_wavefront_obj_save_tri_mesh

    io_wavefront_obj_save_tri_mesh(tri2vtx, vtx2xyz, path_file)
