import math
import numpy
from del_msh_numpy import TriMesh, BVH, Raycast

def test_01():
    tri2vtx, vtx2xyz = TriMesh.capsule()
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


def test_02():
    tri2vtx, vtx2xyz = TriMesh.sphere(0.8, 64, 32)
    bvhnodes = TriMesh.bvhnodes_tri(tri2vtx, vtx2xyz)
    assert bvhnodes.shape[1] == 3
    bvhnode2aabb = BVH.aabb_uniform_mesh(tri2vtx, vtx2xyz, bvhnodes, aabbs=None, vtx2xyz1=None)
    transform_ndc2world = numpy.eye(4, dtype=numpy.float32)
    pix2tri = numpy.ndarray(shape=(300, 300), dtype=numpy.uint64)
    Raycast.pix2tri(
        pix2tri,
        tri2vtx,
        vtx2xyz,
        bvhnodes,
        bvhnode2aabb,
        transform_ndc2world)
    mask = pix2tri != numpy.iinfo(numpy.uint64).max
    num_true = numpy.count_nonzero(mask)
    ratio0 = float(num_true)/float(mask.shape[0]*mask.shape[1])
    ratio1 = 0.8*0.8*numpy.pi*0.25
    assert abs(ratio0-ratio1) < 0.00013
    #
    from PIL import Image
    img = Image.fromarray((mask.astype(numpy.uint8) * 255))
    img.save("../target/hoge.png")
