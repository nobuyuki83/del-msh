import numpy
import numpy.typing

def build_topology(
        vtx2xy: numpy.typing.NDArray):
    if vtx2xy.shape[1] == 2:
        from del_msh.del_msh import kdtree_build_2d
        return kdtree_build_2d(vtx2xy)
    else:
        assert False


def build_edge(
        tree: numpy.typing.NDArray,
        vtx2xy: numpy.typing.NDArray):
    vmin = vtx2xy.min(axis=0)
    vmax = vtx2xy.max(axis=0)
    if vtx2xy.shape[1] == 2:
        from del_msh.del_msh import kdtree_edge_2d
        return kdtree_edge_2d(tree, vtx2xy, vmin, vmax)
    else:
        assert False
