

def vtx2xyz_from_helix(num_vtx, elen, rad, pitch):
    from .del_msh_numpy import polyline_vtx2xyz_from_helix
    return polyline_vtx2xyz_from_helix(num_vtx, elen, rad, pitch)


def vtx2framex_from_vtx2xyz(vtx2xyz):
    from .del_msh_numpy import polyline_vtx2framex_from_vtx2xyz
    return polyline_vtx2framex_from_vtx2xyz(vtx2xyz)


def vtx2vtx_rods(hair2root):
    from .del_msh_numpy import polyline_vtx2vtx_rods
    return polyline_vtx2vtx_rods(hair2root)


def save_wavefront_obj(vtx2xyz, path: str):
    from .del_msh_numpy import polyline_save_wavefront_obj
    polyline_save_wavefront_obj(vtx2xyz, path)
