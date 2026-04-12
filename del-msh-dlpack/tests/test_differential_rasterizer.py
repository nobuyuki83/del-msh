import math
import pathlib
#
import torch
from PIL import Image
#
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.Vtx2Elem.torch
import del_msh_dlpack.Edge2Elem.torch
import del_msh_dlpack.Edge2Vtx.torch as Edge2Vtx
import del_msh_dlpack.Vtx2Vtx.torch
import del_msh_dlpack.Pix2Depth.torch as Pix2Depth
import del_msh_dlpack.DifferentiableAntialias.torch as DifferentiableAntialias
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
    Pix2Tri.update_pix2tri(tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, pix2tri)
    #
    edge2vtx = TriMesh3.make_edge2vtx(tri2vtx, vtx2xyz.shape[0])
    edge2tri = TriMesh3.make_edge2tri(tri2vtx, vtx2xyz.shape[0], edge2vtx)
    cedge2vtx = del_msh_dlpack.Edge2Vtx.torch.contour_for_triangle_mesh(tri2vtx,vtx2xyz,transform_world2ndc,edge2vtx,edge2tri)
    pix2occin = torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0).to(torch.float32)
    #
    pix2occout = DifferentiableAntialias.antialias(
        cedge2vtx,
        vtx2xyz,
        transform_world2pix,
        pix2tri,
        pix2occin
    )
    #
    img = (pix2occout.numpy() * 255).clip(0, 255).astype('uint8')
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
        dldw_pix2occ = torch.zeros((img_h, img_w), dtype=torch.float32)
        dldw_pix2occ.view(-1)[i_pix] = 1.0
        dldw_vtx2xyz = DifferentiableAntialias.bwd_antialias(
            cedge2vtx,
            vtx2xyz,
            transform_world2pix,
            pix2occin,
            dldw_pix2occ,
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
    d_tri2vtx = tri2vtx.cuda()
    d_vtx2xyz = vtx2xyz.cuda()
    #
    transform_ndc2world = transform_world2ndc.inverse()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(d_tri2vtx, d_vtx2xyz)
    assert torch.equal(bvhnodes, d_bvhnodes.cpu())
    assert torch.equal(bvhnode2aabb, d_bvhnode2aabb.cpu())
    #
    pix2tri = torch.full((img_shape[1], img_shape[0]),
                         torch.iinfo(torch.uint32).max, device=torch.device("cpu"), dtype=torch.uint32)
    del_msh_dlpack.Pix2Tri.torch.update_pix2tri(tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, pix2tri)
    d_pix2tri = torch.full((img_shape[1], img_shape[0]),
                         torch.iinfo(torch.uint32).max, device=torch.device("cuda"), dtype=torch.uint32)
    del_msh_dlpack.Pix2Tri.torch.update_pix2tri(d_tri2vtx, d_vtx2xyz, d_bvhnodes, d_bvhnode2aabb, transform_ndc2world.cuda(), d_pix2tri)
    assert torch.equal(pix2tri, d_pix2tri.cpu())
    #
    vtx2idx_offset, idx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)
    d_vtx2idx_offset, d_idx2vtx = del_msh_dlpack.Vtx2Vtx.torch.from_uniform_mesh(d_tri2vtx, vtx2xyz.shape[0], False)
    assert torch.equal(vtx2idx_offset, d_vtx2idx_offset.cpu())
    assert torch.equal(idx2vtx, d_idx2vtx.cpu())
    #
    edge2vtx = torch.empty((idx2vtx.shape[0], 2), dtype=torch.uint32)
    del_msh_dlpack.Edge2Vtx.torch.from_vtx2vtx(vtx2idx_offset, idx2vtx, edge2vtx)
    d_edge2vtx = torch.empty((d_idx2vtx.shape[0], 2), dtype=torch.uint32, device=torch.device("cuda"))
    del_msh_dlpack.Edge2Vtx.torch.from_vtx2vtx(d_vtx2idx_offset, d_idx2vtx, d_edge2vtx)
    assert torch.equal(edge2vtx, d_edge2vtx.cpu())
    #
    vtx2jdx_offset, jdx2tri = del_msh_dlpack.Vtx2Elem.torch.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0])
    d_vtx2jdx_offset, d_jdx2tri = del_msh_dlpack.Vtx2Elem.torch.from_uniform_mesh(d_tri2vtx, vtx2xyz.shape[0])
    #
    num_edge = edge2vtx.shape[0]
    edge2tri = torch.empty((num_edge, 2), dtype=torch.uint32)
    del_msh_dlpack.Edge2Elem.torch.from_edge2vtx_of_tri2vtx_with_vtx2vtx(edge2vtx, tri2vtx, vtx2jdx_offset, jdx2tri, edge2tri)
    d_edge2tri = torch.empty((num_edge, 2), dtype=torch.uint32, device=torch.device("cuda"))
    del_msh_dlpack.Edge2Elem.torch.from_edge2vtx_of_tri2vtx_with_vtx2vtx(d_edge2vtx, d_tri2vtx, d_vtx2jdx_offset, d_jdx2tri, d_edge2tri)
    assert torch.equal(edge2tri, d_edge2tri.cpu())
    #
    cedge2vtx = del_msh_dlpack.Edge2Vtx.torch.contour_for_triangle_mesh(tri2vtx,vtx2xyz,transform_world2ndc,edge2vtx,edge2tri)
    d_cedge2vtx = del_msh_dlpack.Edge2Vtx.torch.contour_for_triangle_mesh(d_tri2vtx,d_vtx2xyz,transform_world2ndc.cuda(),d_edge2vtx,d_edge2tri)
    assert torch.equal(cedge2vtx, d_cedge2vtx.cpu())
    #
    pix2occin = torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0).to(torch.float32)
    pix2occ = DifferentiableAntialias.antialias(
        cedge2vtx,
        vtx2xyz,
        transform_world2pix,
        pix2tri,
        pix2occin,
    )
    d_pix2occin = torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0).to(torch.float32).cuda()
    d_pix2occ = DifferentiableAntialias.antialias(
        d_cedge2vtx,
        d_vtx2xyz,
        transform_world2pix.cuda(),
        d_pix2tri,
        d_pix2occin,
    )
    assert (pix2occ-d_pix2occ.cpu()).abs().max().item() < 0.008
    #
    dldw_pix2occ = torch.rand((img_shape[1], img_shape[0]), dtype=torch.float32)
    dldw_vtx2xyz = DifferentiableAntialias.bwd_antialias(
        cedge2vtx,
        vtx2xyz,
        transform_world2pix,
        pix2occin,
        dldw_pix2occ,
        pix2tri,
    )
    d_dldw_pix2occ = dldw_pix2occ.cuda()
    d_dldw_vtx2xyz = DifferentiableAntialias.bwd_antialias(
        d_cedge2vtx,
        d_vtx2xyz,
        transform_world2pix.cuda(),
        d_pix2occin,
        d_dldw_pix2occ,
        d_pix2tri,
    )
    assert (dldw_vtx2xyz-d_dldw_vtx2xyz.cpu()).abs().max().item() < 1.0e-4


def test_autograd():
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape = example1()
    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    vtx2xyz.requires_grad_(True)
    edge2vtx = TriMesh3.make_edge2vtx(tri2vtx, vtx2xyz.shape[0])
    edge2tri = TriMesh3.make_edge2tri(tri2vtx, vtx2xyz.shape[0], edge2vtx)
    #
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = torch.zeros((img_shape[1], img_shape[0]), dtype=torch.uint32)
    Pix2Tri.update_pix2tri(tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, pix2tri)
    depth = Pix2Depth.Pix2DepthFunction.apply(vtx2xyz, pix2tri, tri2vtx, transform_ndc2world)
    cedge2vtx = Edge2Vtx.contour_for_triangle_mesh(tri2vtx,vtx2xyz,transform_world2ndc,edge2vtx,edge2tri)
    deptha = DifferentiableAntialias.DifferentiableAntialiasFunction.apply(cedge2vtx, vtx2xyz, transform_world2pix, pix2tri, depth)
    trgt = torch.rand((img_shape[1], img_shape[0]), dtype=torch.float32)
    loss = torch.dot(trgt.flatten(), deptha.flatten())
    loss.backward()
    grad = vtx2xyz.grad
    vtx2xyz1 = vtx2xyz + vtx2xyz.grad * 0.1
    if torch.cuda.is_available():
        (d_tri2vtx, d_vtx2xyz) = (tri2vtx.cuda(), vtx2xyz.detach().clone().cuda().requires_grad_(True))
        d_edge2vtx = TriMesh3.make_edge2vtx(d_tri2vtx, d_vtx2xyz.shape[0])
        d_edge2tri = TriMesh3.make_edge2tri(d_tri2vtx, d_vtx2xyz.shape[0], d_edge2vtx)
        d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(d_tri2vtx, d_vtx2xyz)
        d_pix2tri = torch.zeros((img_shape[1], img_shape[0]), dtype=torch.uint32, device=torch.device("cuda"))
        Pix2Tri.update_pix2tri(d_tri2vtx, d_vtx2xyz, d_bvhnodes, d_bvhnode2aabb, transform_ndc2world.cuda(), d_pix2tri)
        d_depth = Pix2Depth.Pix2DepthFunction.apply(d_vtx2xyz, d_pix2tri, d_tri2vtx, transform_ndc2world.cuda())
        d_cedge2vtx = Edge2Vtx.contour_for_triangle_mesh(d_tri2vtx,d_vtx2xyz,transform_world2ndc.cuda(),d_edge2vtx,d_edge2tri)
        d_deptha = DifferentiableAntialias.DifferentiableAntialiasFunction.apply(d_cedge2vtx, d_vtx2xyz, transform_world2pix.cuda(), d_pix2tri, d_depth)
        d_loss = torch.dot(trgt.cuda().flatten(), d_deptha.flatten())
        d_loss.backward()
        d_grad = d_vtx2xyz.grad
        assert( (loss - d_loss.cpu()).abs() < 0.002 )
        assert (d_grad.cpu() - grad).abs().max() < 1.0e-4





