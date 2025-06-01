import numpy
import numpy.typing
from del_msh_numpy import PolyLoop, TriMesh


def do_many_thing_for_polyloop(vtxi2xyi: numpy.typing.NDArray):
    tri2vtx, vtx2xy = PolyLoop.tesselation2d(
            vtxi2xyi, resolution_edge=0.11, resolution_face=-1)
    tri2cc = TriMesh.tri2circumcenter(tri2vtx, vtx2xy)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xy.shape[0])
    edge2mp = (vtx2xy[edge2vtx[:,0]] + vtx2xy[edge2vtx[:,1]])*0.5


def test_0():
    vtxi2xyi = numpy.array([
        [0, 0],
        [1, 0],
        [1, 0.1],
        [0, 0.1]], dtype=numpy.float32)
    do_many_thing_for_polyloop(vtxi2xyi)

