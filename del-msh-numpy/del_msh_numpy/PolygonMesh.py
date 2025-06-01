"""
Functions for polygon mesh (e.g., mixture of triangles, quadrilaterals, pentagons) in 2D and 3D
"""

import numpy
import numpy.typing


def triangles(
        pelem2pidx: numpy.typing.NDArray,
        pidx2vtxxyz: numpy.typing.NDArray) -> numpy.typing.NDArray:
    from .del_msh_numpy import triangles_from_polygon_mesh
    return triangles_from_polygon_mesh(pelem2pidx, pidx2vtxxyz)


def extract(
        elem2idx: numpy.typing.NDArray,
        idx2vtx: numpy.typing.NDArray,
        elem2bool: numpy.typing.NDArray):
    from .del_msh_numpy import extract_flagged_polygonal_element
    return extract_flagged_polygonal_element(elem2idx, idx2vtx, elem2bool)


def edges(
        elem2idx: numpy.typing.NDArray,
        idx2vtx: numpy.typing.NDArray,
        num_vtx: int):
    from .del_msh_numpy import edge2vtx_polygon_mesh
    return edge2vtx_polygon_mesh(elem2idx, idx2vtx, num_vtx)
