import numpy


def tesselation2d(
        vtx2xy,
        resolution_edge=-1.,
        resolution_face=-1.):
    from .del_msh_numpy import tesselation2d
    return tesselation2d(vtx2xy, resolution_edge, resolution_face)



def area2(
        vtx2xy: numpy.typing.NDArray) -> float:
    if vtx2xy.dtype == numpy.float32:
        from .del_msh_numpy import polyloop2_area_f32
        return polyloop2_area_f32(vtx2xy)
    elif vtx2xy.dtype == numpy.float64:
        from .del_msh_numpy import polyloop2_area_f64
        return polyloop2_area_f64(vtx2xy)
