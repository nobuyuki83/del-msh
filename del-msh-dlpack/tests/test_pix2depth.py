import math
import pathlib
#
import torch
from PIL import Image
#
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.Pix2Depth.torch as Pix2Depth
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz

def example1():
    tri2vtx, vtx2xyz = TriMesh3.torus(1.3, 0.4, 64, 32)
    transform0 = Mat44.from_x_rotation(1.15)
    transform1 = Mat44.from_translation(0., 0.3, -4)
    #transform1 = Mat44.from_translation(0., 0.3, 0)
    transform = transform1 @ transform0
    vtx2xyz = Vtx2Xyz.transform_affine(vtx2xyz, transform)
    transform_world2ndc = Mat44.camera_perspective_blender(1.0, 30.0, 2.,  6.0, True)
    #transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
    img_shape = (128, 128)
    return tri2vtx, vtx2xyz, transform_world2ndc, img_shape


def test_cpu_cuda_match_with_autograd():
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape = example1()

    vtx2xyz.requires_grad = True
    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    ##
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = torch.empty((img_shape[1], img_shape[0]), dtype=torch.uint32)
    pix2tri.fill_(torch.iinfo(torch.uint32).max)
    Pix2Tri.update_pix2tri(tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, pix2tri)

    pix2depth = Pix2Depth.Pix2DepthFunction.apply(vtx2xyz, pix2tri, tri2vtx, transform_ndc2world)
    img = (pix2depth.detach().numpy() * 255).clip(0, 255).astype('uint8')
    path0 = pathlib.Path(__file__).parent.parent.parent / "target" / "del_msh_dlpack__pix2depth1.png"
    path0.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path0)

    dldw_depth = torch.rand(size=img_shape)
    loss = torch.sum(dldw_depth * pix2depth)

    print(loss.item())
    loss.backward()
    dldw_vtx2xyz = vtx2xyz.grad

    if torch.cuda.is_available():
        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.cuda().detach()
        d_transform_ndc2world = transform_ndc2world.cuda()
        d_dldw_depth = dldw_depth.cuda()
        #
        d_vtx2xyz.requires_grad = True
        d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(d_tri2vtx, d_vtx2xyz)
        d_pix2tri = torch.empty((img_shape[1], img_shape[0]), dtype=torch.uint32, device='cuda')
        d_pix2tri.fill_(torch.iinfo(torch.uint32).max)
        Pix2Tri.update_pix2tri(d_tri2vtx, d_vtx2xyz, d_bvhnodes, d_bvhnode2aabb, d_transform_ndc2world, d_pix2tri)
        assert torch.equal( d_pix2tri.cpu(), pix2tri )
        d_pix2depth = Pix2Depth.Pix2DepthFunction.apply(d_vtx2xyz, d_pix2tri, d_tri2vtx, d_transform_ndc2world)
        assert (d_pix2depth.cpu()-pix2depth).norm().item() < 1.0e-5
        d_loss = torch.sum(d_dldw_depth * d_pix2depth)
        d_loss.backward()
        d_dldw_vtx2xyz = d_vtx2xyz.grad
        assert (d_dldw_vtx2xyz.cpu()-dldw_vtx2xyz).max() < 5.0e-4


