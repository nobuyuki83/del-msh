

def load_cfd_mesh(path_file: str):
    from ..del_msh_dlpack import io_cfd_mesh_txt_load

    return io_cfd_mesh_txt_load(path_file)