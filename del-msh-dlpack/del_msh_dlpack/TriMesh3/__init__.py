


def tri2normal(tri2vtx, vtx2xyz, tri2normal):
    from ..del_msh_dlpack import trimesh3_tri2normal
    trimesh3_tri2normal(tri2vtx, vtx2xyz, tri2normal)


def bwd_tri2normal(tri2vtx, vtx2xyz, dw_tri2normal, dw_vtx2xyz):
    from ..del_msh_dlpack import trimesh3_bwd_tri2normal
    trimesh3_bwd_tri2normal(tri2vtx, vtx2xyz, dw_tri2normal, dw_vtx2xyz)