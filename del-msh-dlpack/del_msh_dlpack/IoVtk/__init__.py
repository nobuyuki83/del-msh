def write_mix_mesh(
        path_file: str,
        vtx2xyz,
        tet2vtx,
        pyrmd2vtx,
        prism2vtx,
        hex2vtx):
    from ..del_msh_dlpack import io_vtk_write_mix_mesh

    return io_vtk_write_mix_mesh(path_file, vtx2xyz, tet2vtx, pyrmd2vtx, prism2vtx, hex2vtx)
