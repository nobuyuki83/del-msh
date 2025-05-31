import math
import numpy
from del_msh import PolyLoop, TriMesh


def test_01():
    vtx2xy_in = numpy.array([
        [0, 0],
        [1, 0],
        [1, 0.6],
        [0.6, 0.6],
        [0.6, 1.0],
        [0, 1]], dtype=numpy.float32)
    ##
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(vtx2xy_in)
    area0 = PolyLoop.area2(vtx2xy_in)
    area1 = TriMesh.tri2area(tri2vtx, vtx2xy).sum()
    assert math.fabs(area0 - area1) < 1.0e-5
    area0 = PolyLoop.area2(vtx2xy_in.astype(numpy.float64))
