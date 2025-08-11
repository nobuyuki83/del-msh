
def test_01():
    from del_msh_numpy import Polyline
    vtx2xyz = Polyline.vtx2xyz_from_helix(30, 0.2, 0.2, 0.5)
    vtx2framex = Polyline.vtx2framex_from_vtx2xyz(vtx2xyz)
    Polyline.save_wavefront_obj(vtx2xyz, "../target/numpy_save_polyline.obj")