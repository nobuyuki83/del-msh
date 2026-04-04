import math
import pathlib
#
import torch
from PIL import Image
#
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.TriMesh3Raycast.torch
import del_msh_dlpack.Vtx2Elem.torch
import del_msh_dlpack.Edge2Elem.torch
import del_msh_dlpack.Edge2Vtx.torch
import del_msh_dlpack.Vtx2Vtx.torch
import del_msh_dlpack.DifferentialRasterizer.torch as DifferentialRasterizer
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz

def apply_colormap_bwr(val: float, vmin: float, vmax: float):
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    if t < 0.5:
        return [int(2*t*255), int(2*t*255), 255]
    else:
        return [255, int((2-2*t)*255), int((2-2*t)*255)]

def example1():
    tri2vtx, vtx2xyz = TriMesh3.torus(1.3, 0.4, 64, 32)
    transform0 = Mat44.from_x_rotation(1.15)
    transform1 = Mat44.from_translation(0., 0.6, 0.)
    transform = transform1 @ transform0
    vtx2xyz = Vtx2Xyz.transform_affine(vtx2xyz, transform)
    transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
    img_shape = (128, 128)
    return tri2vtx, vtx2xyz, transform_world2ndc, img_shape


def test_save_fwd_diff_image():
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape = example1()
    transform_ndc2world = transform_world2ndc.inverse()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    ##
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = torch.empty((img_shape[1], img_shape[0]), dtype=torch.uint32)
    pix2tri.fill_(torch.iinfo(torch.uint32).max)
    del_msh_dlpack.TriMesh3Raycast.torch.update_pix2tri(tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, pix2tri)
    #
    vtx2idx_offset, idx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)
    edge2vtx = torch.empty((idx2vtx.shape[0], 2), dtype=torch.uint32)
    del_msh_dlpack.Edge2Vtx.torch.from_vtx2vtx(vtx2idx_offset, idx2vtx, edge2vtx)
    num_edge = edge2vtx.shape[0]
    #
    vtx2jdx_offset, jdx2tri = del_msh_dlpack.Vtx2Elem.torch.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0])
    #
    edge2tri = torch.empty((num_edge, 2), dtype=torch.uint32)
    del_msh_dlpack.Edge2Elem.torch.from_edge2vtx_of_tri2vtx_with_vtx2vtx(edge2vtx, tri2vtx, vtx2jdx_offset, jdx2tri, edge2tri)
    #
    edge2vtx_contour = del_msh_dlpack.Edge2Vtx.torch.contour_for_triangle_mesh(tri2vtx,vtx2xyz,transform_world2ndc,edge2vtx,edge2tri)
    img_data = torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0).to(torch.float32)
    #
    DifferentialRasterizer.antialias(
        edge2vtx_contour,
        vtx2xyz,
        transform_world2pix,
        pix2tri,
        img_data,
    )
    #
    img = (img_data.numpy() * 255).clip(0, 255).astype('uint8')
    path0 = pathlib.Path(__file__).parent.parent.parent / "target" / "del_msh_dlpack__antialias.png"
    path0.parent.mkdir(exist_ok=True)
    Image.fromarray(img, mode="L").save(path0)
    #
    # gradient visualization: d(pixel) / d(vtx) projected onto x-direction
    num_vtx = vtx2xyz.shape[0]
    dxyz = torch.zeros_like(vtx2xyz)
    dxyz[:, 0] = 1.0  # x-direction perturbation
    #
    img_h, img_w = img_shape[1], img_shape[0]
    num_pix = img_h * img_w
    pix2rgb_diff = torch.zeros((img_h, img_w, 3), dtype=torch.uint8)
    vmin, vmax = -float(img_w), float(img_w)
    for i_pix in range(num_pix):
        dldw_pixval = torch.zeros((img_h, img_w), dtype=torch.float32)
        dldw_pixval.view(-1)[i_pix] = 1.0
        dldw_vtx2xyz = torch.zeros((num_vtx, 3), dtype=torch.float32)
        DifferentialRasterizer.bwd_antialias(
            edge2vtx_contour,
            vtx2xyz,
            dldw_vtx2xyz,
            transform_world2pix,
            dldw_pixval,
            pix2tri,
        )
        dpix = (dxyz * dldw_vtx2xyz).sum().item()
        c = apply_colormap_bwr(dpix, vmin, vmax)
        pix2rgb_diff.view(-1, 3)[i_pix] = torch.tensor(c, dtype=torch.uint8)
    #
    path1 = pathlib.Path(__file__).parent.parent.parent / "target" / "del_msh_dlpack__antialias_diff.png"
    Image.fromarray(pix2rgb_diff.numpy(), mode="RGB").save(path1)



def test_match_cpu_cuda():
    if not torch.cuda.is_available():
        print("cuda is not available, skip test")
        return
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape = example1()
    transform_ndc2world = transform_world2ndc.inverse()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    ##
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)

    d_tri2vtx = tri2vtx.cuda()
    d_vtx2xyz = vtx2xyz.cuda()
    d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(d_tri2vtx, d_vtx2xyz)





