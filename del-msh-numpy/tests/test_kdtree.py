import numpy


def test_01():
    vtx2xy = numpy.random.rand(100,2)
    from del_msh_numpy.del_msh_numpy import kdtree_build_2d
    tree = kdtree_build_2d(vtx2xy)
    from del_msh_numpy.del_msh_numpy import kdtree_edge_2d
    vmin = vtx2xy.min(axis=0)
    vmax = vtx2xy.max(axis=0)
    edge2node2xy = kdtree_edge_2d(tree, vtx2xy, vmin, vmax)