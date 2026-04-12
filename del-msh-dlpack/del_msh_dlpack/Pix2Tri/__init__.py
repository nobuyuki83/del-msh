def update_pix2tri(
        tri2vtx,
        vtx2xyz,
        bvhnodes,
        bvhnode2aabb,
        transform_ndc2world,
        pix2tri,
        stream_ptr=0):
    from ..del_msh_dlpack import trimesh3_raycast_update_pix2tri

    trimesh3_raycast_update_pix2tri(
        pix2tri,
        tri2vtx,
        vtx2xyz,
        bvhnodes,
        bvhnode2aabb,
        transform_ndc2world,
        stream_ptr
    )
