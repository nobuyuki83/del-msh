import math
#
import numpy
#
from del_msh import TriMesh, BVH


def test_tri_self_intersection():
    tri2vtx, vtx2xyz = TriMesh.sphere(1., 8, 4)
    bvhnodes = TriMesh.bvhnodes_tri(tri2vtx, vtx2xyz)
    assert bvhnodes.shape[1] == 3
    aabbs = BVH.aabb_uniform_mesh(tri2vtx, vtx2xyz, bvhnodes)
    assert bvhnodes.shape[0] == aabbs.shape[0]
    assert aabbs.shape[1] == 6
    assert numpy.linalg.norm(aabbs[0] - numpy.array([-1., -1., -1., 1., 1., 1])) < 1.0e-5
    edge2node2xyz0, edge2tri0 = TriMesh.self_intersection(tri2vtx, vtx2xyz)
    edge2node2xyz, edge2tri = TriMesh.self_intersection(tri2vtx, vtx2xyz, bvhnodes, aabbs)
    assert edge2node2xyz0.shape == edge2node2xyz.shape
    assert edge2node2xyz.shape[0] == 0


def test_ccd():
    tri2vtx, vtx2xyz0 = TriMesh.sphere(1.0, ndiv_longtitude=16, ndiv_latitude=32)
    edge2vtx = TriMesh.edge2vtx(tri2vtx, vtx2xyz0.shape[0])
    # print(edge2vtx)
    vtx2uvw = numpy.zeros_like(vtx2xyz0)
    vtx2uvw[:, 0] += - 2 * numpy.power(vtx2xyz0[:, 0], 5)
    vtx2xyz1 = vtx2xyz0 + 1.0 * vtx2uvw
    bvhnodes, roots = TriMesh.bvhnodes_vtxedgetri(edge2vtx, tri2vtx, vtx2xyz0)
    aabbs = TriMesh.aabb_vtxedgetri(edge2vtx, tri2vtx, vtx2xyz0, bvhnodes, roots, vtx2xyz1=vtx2xyz1)
    pairs0, times0 = TriMesh.ccd_intersection_time(edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1)
    pairs, times = TriMesh.ccd_intersection_time(edge2vtx, tri2vtx, vtx2xyz0, vtx2xyz1, bvhnodes, aabbs, roots)
    assert pairs0.shape == pairs.shape
    intersecting_time = numpy.min(times)
    assert math.fabs(intersecting_time-0.5) < 1.0e-5
    #
    vtx2xyz1 = vtx2xyz0 + intersecting_time * 0.999 * vtx2uvw
    edge2node2xyz0, edge2tri0 = TriMesh.self_intersection(tri2vtx, vtx2xyz1)
    edge2node2xyz, edge2tri = TriMesh.self_intersection(tri2vtx, vtx2xyz1, bvhnodes, aabbs, roots[2])
    assert edge2node2xyz0.shape == edge2node2xyz.shape
    assert edge2node2xyz.shape[0] == 0
    #
    vtx2xyz1 = vtx2xyz0 + intersecting_time * 1.001 * vtx2uvw
    edge2node2xyz0, edge2tri0 = TriMesh.self_intersection(tri2vtx, vtx2xyz1)
    edge2node2xyz, edge2tri = TriMesh.self_intersection(tri2vtx, vtx2xyz1, bvhnodes, aabbs, roots[2])
    assert edge2node2xyz0.shape == edge2node2xyz.shape
    assert edge2node2xyz.shape[0] != 0