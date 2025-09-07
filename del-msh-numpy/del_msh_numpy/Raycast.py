import numpy

def pix2tri(pix2tri, tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world):
    assert pix2tri.dtype == numpy.uint64
    assert tri2vtx.dtype == numpy.uint64
    assert vtx2xyz.dtype == numpy.float32
    assert bvhnodes.dtype == numpy.uint64
    assert bvhnode2aabb.dtype == numpy.float32
    assert transform_ndc2world.dtype == numpy.float32
    #
    from .del_msh_numpy import trimesh3_raycast_update_pix2tri
    trimesh3_raycast_update_pix2tri(
        pix2tri,
        tri2vtx,
        vtx2xyz,
        bvhnodes,
        bvhnode2aabb,
        transform_ndc2world.ravel(order='F'))


def pix2depth(pix2depth, tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world):
    assert pix2depth.dtype == numpy.float32
    assert tri2vtx.dtype == numpy.uint64
    assert vtx2xyz.dtype == numpy.float32
    assert bvhnodes.dtype == numpy.uint64
    assert bvhnode2aabb.dtype == numpy.float32
    assert transform_ndc2world.dtype == numpy.float32
    #
    from .del_msh_numpy import trimesh3_raycast_render_depth_bvh
    trimesh3_raycast_render_depth_bvh(
        pix2depth,
        transform_ndc2world.ravel(order='F'),
        tri2vtx,
        vtx2xyz,
        bvhnodes,
        bvhnode2aabb)
