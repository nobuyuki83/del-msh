def load_tri_mesh(path: str):
    from ..del_msh_dlpack import io_off_load_tri_mesh

    return io_off_load_tri_mesh(path)
