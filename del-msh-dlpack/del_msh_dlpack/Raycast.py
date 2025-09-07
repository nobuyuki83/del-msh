import numpy

def pix2tri(pix2tri, tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world):
    from .del_msh_dlpack import trimesh3_raycast_update_pix2tri
    trimesh3_raycast_update_pix2tri(
        pix2tri.__dlpack__(),
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        bvhnodes.__dlpack__(),
        bvhnode2aabb.__dlpack__(),
        transform_ndc2world.ravel(order='F').__dlpack__())