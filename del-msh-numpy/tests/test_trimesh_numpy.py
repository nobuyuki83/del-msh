import math
import numpy
from del_msh_numpy import TriMesh, BVH, Raycast

def test_01():
    tri2vtx, vtx2xyz = TriMesh.capsule()
    #
    TriMesh.save_wavefront_obj(tri2vtx, vtx2xyz, "../target/hogehoge.obj")
    tri2vtx1, vtx2xyz1 = TriMesh.load_wavefront_obj("../target/hogehoge.obj")
    assert tri2vtx.shape == tri2vtx1.shape
    assert vtx2xyz.shape == vtx2xyz1.shape
    #
    tri2vtx, vtx2xyz = TriMesh.torus()
    tri2vtx, vtx2xyz = TriMesh.sphere()
    tri2node2xyz = TriMesh.unindexing(tri2vtx, vtx2xyz)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz.shape[0])
    vtx2idx, idx2vtx = TriMesh.vtx2vtx(tri2vtx, vtx2xyz.shape[0])
    tri2tri = TriMesh.tri2tri(tri2vtx, vtx2xyz.shape[0])
    tri2dist = TriMesh.tri2distance(0, tri2tri)
    assert tri2dist[0] == 0
    areas = TriMesh.tri2area(tri2vtx, vtx2xyz)
    assert math.fabs(areas.sum() - 4. * math.pi) < 0.1
    cumsum_areas = numpy.cumsum(numpy.append(numpy.zeros(1, dtype=numpy.float32), areas))
    sample = TriMesh.sample(cumsum_areas, 0.5, 0.1)
    samples2xyz = TriMesh.sample_many(tri2vtx, vtx2xyz, num_sample=1000)

