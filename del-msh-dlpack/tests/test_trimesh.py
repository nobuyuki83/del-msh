import math
import os.path

import numpy
import del_msh_numpy.TriMesh
import del_msh_numpy.BVH
import del_msh_dlpack.Raycast
import del_msh_dlpack.TriMesh3.np

def test_01():
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.sphere(0.8, 128, 64)
    tri2normal = del_msh_dlpack.TriMesh3.np.tri2normal(tri2vtx.astype(numpy.uint32), vtx2xyz)
    tri2area = numpy.linalg.norm(tri2normal, axis=1)
    area = tri2area.sum() * 0.5
    area0 = 0.8 * 0.8 * 4.0 * numpy.pi
    assert abs(area-area0)/area0  < 0.001


def test_02():
    '''
    test raycast
    '''
    tri2vtx, vtx2xyz = del_msh_numpy.TriMesh.sphere(0.8, 64, 32)
    bvhnodes = del_msh_numpy.TriMesh.bvhnodes_tri(tri2vtx, vtx2xyz)
    assert bvhnodes.shape[1] == 3
    bvhnode2aabb = del_msh_numpy.BVH.aabb_uniform_mesh(tri2vtx, vtx2xyz, bvhnodes, aabbs=None, vtx2xyz1=None)
    assert bvhnode2aabb.shape[1] == 6
    transform_ndc2world = numpy.eye(4, dtype=numpy.float32)
    #
    pix2tri = numpy.ndarray(shape=(300, 300), dtype=numpy.uint64)

    del_msh_dlpack.Raycast.pix2tri(
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

    from pathlib import Path
    path = Path(__file__).resolve().parent.parent.parent
    print(path.resolve() )

    from PIL import Image
    img = Image.fromarray((mask.astype(numpy.uint8) * 255))
    file_path = path /  'target/del_msh_delpack__pix2tri.png'
    img.save(file_path)
    #

    '''
    pix2depth = numpy.zeros(shape=(300, 300), dtype=numpy.float32)
    Raycast.pix2depth(
        pix2depth,
        tri2vtx=tri2vtx,
        vtx2xyz=vtx2xyz,
        bvhnodes=bvhnodes,
        bvhnode2aabb=bvhnode2aabb,
        transform_ndc2world = transform_ndc2world)
    img = Image.fromarray((pix2depth * 255.).astype(numpy.uint8))
    img.save("../target/del_msh_numpy__pix2depth.png")
    '''



