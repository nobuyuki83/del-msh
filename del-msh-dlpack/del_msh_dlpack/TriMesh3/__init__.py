def tri2normal(tri2vtx, vtx2xyz, tri2normal, stream_ptr=0):
    from ..del_msh_dlpack import trimesh3_tri2normal

    trimesh3_tri2normal(tri2vtx, vtx2xyz, tri2normal, stream_ptr)


def bwd_tri2normal(tri2vtx, vtx2xyz, dw_tri2normal, dw_vtx2xyz, stream_ptr=0):
    from ..del_msh_dlpack import trimesh3_bwd_tri2normal

    trimesh3_bwd_tri2normal(tri2vtx, vtx2xyz, dw_tri2normal, dw_vtx2xyz, stream_ptr)


def load_nastran(
        path_file: str):
    from ..del_msh_dlpack import io_nastran_load_tri_mesh

    return io_nastran_load_tri_mesh(path_file)


def save_wavefront_obj(tri2vtx, vtx2xyz, path_file):
    from ..del_msh_dlpack import io_wavefront_obj_save_tri_mesh

    io_wavefront_obj_save_tri_mesh(tri2vtx, vtx2xyz, path_file)


def torus(major_raidus: float, minor_radius: float, ndiv_major: int, ndiv_minor: int):
    from ..del_msh_dlpack import trimesh3_primitive_torus_zup

    return trimesh3_primitive_torus_zup(major_raidus, minor_radius, ndiv_major, ndiv_minor)


def sphere(raidus: float, ndiv_longtitude: int, ndiv_latitude: int):
    from ..del_msh_dlpack import trimesh3_primitive_sphere_yup

    return trimesh3_primitive_sphere_yup(raidus, ndiv_longtitude, ndiv_latitude)


def aabb_from_bvhnodes(tri2vtx, vtx2xyz0, vtx2xyz1, bvhnodes, bvhnode2aabb):
    from ..del_msh_dlpack import bvhnode2aabb_update_aabb

    bvhnode2aabb_update_aabb(
        bvhnode2aabb, 0, bvhnodes, tri2vtx, vtx2xyz0, vtx2xyz1)