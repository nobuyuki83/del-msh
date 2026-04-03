import numpy as np
#
from .. import util_numpy

def update_pix2tri(
        tri2vtx: np.ndarray,
        vtx2xyz: np.ndarray,
        bvhnodes: np.ndarray,
        bvhnode2aabb: np.ndarray,
        transform_ndc2world: np.ndarray,
        pix2tri: np.ndarray):
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    num_bvhnode = bvhnodes.shape[0]
    img_w = pix2tri.shape[1]
    img_h = pix2tri.shape[0]
    #
    print(num_bvhnode, num_tri, num_vtx)
    assert num_bvhnode == num_tri * 2 - 1
    util_numpy.assert_shape_dtype(tri2vtx, (num_tri,3), np.uint32)
    util_torch.assert_shape_dtype(vtx2xyz, (num_vtx, 3), np.float32)
    util_torch.assert_shape_dtype(bvhnodes, (num_bvhnode, 3), np.uint32)
    util_torch.assert_shape_dtype(bvhnode2aabb, (num_bvhnode, 6), np.float32)
    util_torch.assert_shape_dtype(pix2tri, (img_h, img_w), np.uint32)
    #
    from ..TriMesh3Raycast import update_pix2tri

    update_pix2tri(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        bvhnodes.__dlpack__(),
        bvhnode2aabb.__dlpack__(),
        transform_ndc2world.T.contiguous().__dlpack__(),
        pix2tri.__dlpack__(),
    )