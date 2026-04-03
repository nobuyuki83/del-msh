import math
import pathlib
#
import torch
#
import del_msh_dlpack.TriMesh3.torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.TriMesh3Raycast.torch


def mat44_from_x_rotation(angle: float, device=None, dtype=torch.float32) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor([
        [1,  0,  0,  0],
        [0,  c, -s,  0],
        [0,  s,  c,  0],
        [0,  0,  0,  1],
    ], device=device, dtype=dtype)

def mat44_from_translation(tx: float, ty: float, tz: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1],
    ], device=device, dtype=dtype)

def mat44_from_scale(sx: float, sy: float, sz: float, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0,  1],
    ], device=device, dtype=dtype)




'''
transformation converting normalized device coordinate (NDC) `[-1,+1]^3` to pixel coordinate
depth (-1, +1) is transformed to (0, +1)
for example:
     [-1,-1,-1] becomes (0, H, 0)
     [+1,+1,+1] becomes (W, 0, 1)

 * Arguments
    * `image_shape` - (width, height)
'''
def mat44_from_transform_ndc2pix(img_shape, device=None, dtype=torch.float32):
    return torch.tensor(
        [
            [
                0.5 * float(img_shape[0]),
                0,
                0,
                0.5 * float(img_shape[0]),
             ],
            [
                0,
                -0.5 * float(img_shape[1]),
                0,
                0.5 * float(img_shape[1]),
            ],
            [
                0,
                0,
                0.5,
                0.5,
            ],
            [
                0,
                0,
                0,
                1,
            ]

        ], device=device, dtype=dtype)



def test0():
    tri2vtx, vtx2xyz = del_msh_dlpack.TriMesh3.torch.torus(1.3, 0.4, 64, 32)
    transform0 = mat44_from_x_rotation(1.15)
    transform1 = mat44_from_translation(0., 0.6, 0.)
    transform = transform1 @ transform0
    ptcpu_vtx2xyzw = torch.hstack([vtx2xyz, torch.ones((vtx2xyz.shape[0], 1))])
    vtx2xyz = (transform @ ptcpu_vtx2xyzw.t()).t()[:, :3].contiguous()
    transform_world2ndc = mat44_from_scale(0.5, 0.5, 0.5)
    transform_ndc2world = transform_world2ndc.inverse()
    img_shape = (128, 128)
    transform_ndc2pix = mat44_from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    ##
    bvhnodes, bvhnode2aabb = del_msh_dlpack.TriMesh3.torch.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = torch.empty((img_shape[1], img_shape[0]), dtype=torch.uint32)
    pix2tri.fill_(torch.iinfo(torch.uint32).max)
    del_msh_dlpack.TriMesh3Raycast.torch.update_pix2tri(tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, pix2tri)
    #
    '''
    img = torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0, 255).to(torch.uint8).numpy()
    path0 = pathlib.Path(__file__).parent.parent.parent / "target" / "diffrenn.png"
    from PIL import Image
    Image.fromarray(img, mode="L").save(path0)
    '''
    #
    from del_msh_dlpack.Vtx2Vtx.torch import from_uniform_mesh
    vtx2idx_offset, idx2vtx = from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)
    edge2vtx = torch.empty((idx2vtx.shape[0], 2), dtype=torch.uint32)
    from del_msh_dlpack.Edge2Vtx.torch import from_vtx2vtx
    from_vtx2vtx(vtx2idx_offset, idx2vtx, edge2vtx)
    num_edge = edge2vtx.shape[0]
    edge2tri = torch.empty((num_edge, 2), dtype=torch.uint32)
    from del_msh_dlpack.Edge2Elem.torch import from_edge2vtx_of_tri2vtx_with_vtx2vtx
    from_edge2vtx_of_tri2vtx_with_vtx2vtx(edge2vtx, tri2vtx, vtx2idx_offset, idx2vtx, edge2tri)






