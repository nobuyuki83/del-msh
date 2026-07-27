import pathlib


from PIL import Image
import torch
import numpy as np

import del_msh_dlpack.EdgeGrad.torch as RasterizedEdgeGradient
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz
import del_msh_dlpack.IoVtk.torch as IoVtk
import del_msh_dlpack.Vtx2Vtx.torch as Vtx2Vtx


def test_gradient_visualization_silhouette():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    from test_differentiable_antialias import apply_colormap_bwr
    from test_pix2depth import example1

    tri2vtx, vtx2xyz, transform_world2ndc, img_shape = example1()
    transform_ndc2world = transform_world2ndc.inverse()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = Pix2Tri.by_raycasting(
        tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
    )
    pix2occ = (
        torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
        .to(torch.float32)
        .unsqueeze(-1)
    )
    # gradient visualization: d(pixel) / d(vtx) projected onto x-direction
    dxyz = torch.zeros_like(vtx2xyz)
    dxyz[:, 0] = 1.0  # x-direction perturbation
    #
    img_h, img_w = img_shape[1], img_shape[0]
    num_pix = img_h * img_w
    pix2rgb_diff = torch.zeros((img_h, img_w, 3), dtype=torch.uint8)
    vmin, vmax = -float(img_w), float(img_w)
    for i_pix in range(num_pix):
        dldw_pix2occ = torch.zeros((img_h, img_w, 1), dtype=torch.float32)
        dldw_pix2occ.view(-1)[i_pix] = 1.0
        dldw_vtx2xyz = RasterizedEdgeGradient.bwd(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ, dldw_pix2occ
        )
        dpix = (dxyz * dldw_vtx2xyz).sum().item()
        c = apply_colormap_bwr(dpix, vmin, vmax)
        pix2rgb_diff.view(-1, 3)[i_pix] = torch.tensor(c, dtype=torch.uint8)
    #
    Image.fromarray(pix2rgb_diff.numpy()).save(
        path_dir / "diff_rasterized_edge_gradient.png"
    )


def test_match_cpu_gpu_microedge_bwd():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    tri2vtx, vtx2xyz = TriMesh3.torus(1.3, 0.4, 64, 32)
    transform0 = Mat44.from_x_rotation(1.15)
    transform1 = Mat44.from_translation(0.0, 0.3, -4)
    # transform1 = Mat44.from_translation(0., 0.3, 0)
    transform = transform1 @ transform0
    vtx2xyz = Vtx2Xyz.transform_homography(vtx2xyz, transform)
    transform_world2ndc = Mat44.camera_perspective_blender(1.0, 30.0, 2.0, 6.0, True)
    # transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
    img_shape = (128, 128)
    #
    transform_ndc2world = transform_world2ndc.inverse()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = Pix2Tri.by_raycasting(
        tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
    )
    pix2occ = (
        torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
        .to(torch.float32)
        .unsqueeze(-1)
    )
    img = (pix2occ.squeeze().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "microedge_bwd.png")
    #
    torch.random.manual_seed(0)
    dldw_pix2occ = torch.rand_like(pix2occ)
    dldw_vtx2xyz = RasterizedEdgeGradient.bwd(
        tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ, dldw_pix2occ
    )

    if torch.cuda.is_available():
        d_dldw_vtx2xyz = RasterizedEdgeGradient.bwd(
            tri2vtx.cuda(),
            vtx2xyz.cuda(),
            transform_world2pix.cuda(),
            pix2tri.cuda(),
            pix2occ.cuda(),
            dldw_pix2occ.cuda(),
        )
        assert (d_dldw_vtx2xyz.cpu() - dldw_vtx2xyz).abs().max() < 2.0e-5



