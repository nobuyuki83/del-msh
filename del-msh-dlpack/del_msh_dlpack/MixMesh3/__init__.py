

def load_cfd_mesh(path_file: str):
    from ..del_msh_dlpack import io_cfd_mesh_txt_load

    return io_cfd_mesh_txt_load(path_file)


def save_vtk(
    path_file: str,
    vtx2xyz,
    tet2vtx,
    pyrmd2vtx,
    prism2vtx):
    from ..del_msh_dlpack import io_vtk_write_mix_mesh

    return io_vtk_write_mix_mesh(path_file, vtx2xyz, tet2vtx, pyrmd2vtx, prism2vtx)


def to_polyhedron_mesh(
    tet2vtx,
    pyrmd2vtx,
    prism2vtx,
    elem2idx_offset,
    idx2vtx):
    from ..del_msh_dlpack import mix_mesh_to_polyhedron_mesh

    return mix_mesh_to_polyhedron_mesh(tet2vtx, pyrmd2vtx, prism2vtx, elem2idx_offset, idx2vtx)


