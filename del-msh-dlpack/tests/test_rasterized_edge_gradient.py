import pathlib


from PIL import Image
import torch

import del_msh_dlpack.RasterizedEdgeGradient.torch as RasterizedEdgeGradient
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz
import del_msh_dlpack.IoVtk.torch as IoVtk



def test_gradient_visualization_silhouette():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    from test_differentiable_antialias import example1, apply_colormap_bwr

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
        path_dir / "diff_rasterized_edge_gradient.png")


def test_match_cpu_gpu_microedge_bwd():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    from test_differentiable_antialias import example1, apply_colormap_bwr

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
            dldw_pix2occ.cuda()
        )
        assert (d_dldw_vtx2xyz.cpu()-dldw_vtx2xyz).abs().max() < 5.0e-6




def example2():
    tri2vtx, vtx2xyz = TriMesh3.sphere(1.0, 64, 32)
    transform1 = Mat44.from_translation(0.0, 0.0, 0.0)
    vtx2xyz = Vtx2Xyz.transform_affine(vtx2xyz, transform1)  # move up
    transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
    img_shape = (128, 128)
    radius = 50.0  # adjustable
    cy, cx = img_shape[1] / 2.0, img_shape[0] / 2.0
    ys = torch.arange(img_shape[1], dtype=torch.float32) + 0.5
    xs = torch.arange(img_shape[0], dtype=torch.float32) + 0.5
    dist2 = (ys.unsqueeze(1) - cy) ** 2 + (xs.unsqueeze(0) - cx) ** 2
    pix2occ_target = (
        torch.where(dist2 <= radius**2, 1.0, 0.0).to(torch.float32).unsqueeze(-1)
    )
    return tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2occ_target


def test_smooth_gradient_staggered_grid():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)

    tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2occ_trg = example2()
    transform_ndc2world = transform_world2ndc.inverse()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = Pix2Tri.by_raycasting(
        tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
    )
    #
    pix2occ_src = (
        torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
        .to(torch.float32)
        .unsqueeze(-1)
    )
    img = (pix2occ_src.squeeze().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "pix2occ_src.png")
    #
    img = (pix2occ_trg.squeeze().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "pix2occ_trg.png")
    #
    dldw_pix2val = pix2occ_src - pix2occ_trg
    hedge2type, hedge2dldr, vedge2type, vedge2dldr = (
        RasterizedEdgeGradient.edge_gradient_and_type(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ_src, dldw_pix2val
        )
    )

    d_hedge2type, d_vedge2type, d_hedge2dldr, d_vedge2dldr = None, None, None, None
    if torch.cuda.is_available():
        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.detach().clone().cuda()
        d_transform_world2pix = transform_world2pix.cuda()
        d_pix2tri = pix2tri.cuda()
        d_pix2occ_src = pix2occ_src.cuda()
        d_dldw_pix2val = dldw_pix2val.cuda()
        d_hedge2type, d_hedge2dldr, d_vedge2type, d_vedge2dldr = (
            RasterizedEdgeGradient.edge_gradient_and_type(
                d_tri2vtx,
                d_vtx2xyz,
                d_transform_world2pix,
                d_pix2tri,
                d_pix2occ_src,
                d_dldw_pix2val
            )
        )
        torch.equal(d_hedge2type.cpu(), hedge2type)
        assert (d_hedge2dldr.cpu()-hedge2dldr).abs().max() < 1.0e-8
        torch.equal(d_vedge2type.cpu(), vedge2type)
        assert (d_vedge2dldr.cpu()-vedge2dldr).abs().max() < 1.0e-8

    num_itr = 1000

    RasterizedEdgeGradient.smooth_gradient(
        hedge2type, vedge2type,  num_itr, hedge2dldr, vedge2dldr,
    )

    IoVtk.write_velocity_on_staggered_grid(
        str(path_dir / "velocity_on_staggered_grid.vtk"),
        hedge2dldr, vedge2dldr)

    img_h, img_w = img_shape[1], img_shape[0]
    ys = torch.arange(img_h, dtype=torch.float32) + 0.5  # (img_h,)
    xs = torch.arange(img_w, dtype=torch.float32) + 0.5  # (img_w,)
    pix2xy = torch.stack(
        torch.meshgrid(xs, ys, indexing="xy"), dim=-1
    )  # (img_h, img_w, 2): [..., 0]=x, [..., 1]=y
    pix2vxvy = RasterizedEdgeGradient.interpolate(
        hedge2dldr, vedge2dldr, pix2xy.reshape(-1, 2)
    )
    pix2xyz = torch.cat(
        [
            pix2xy.reshape(-1, 2),
            torch.zeros(img_h * img_w, 1, dtype=torch.float32),
        ],
        dim=1,
    )
    pix2vxvyvz = torch.cat(
        [
            pix2vxvy.reshape(-1, 2),
            torch.zeros(img_h * img_w, 1, dtype=torch.float32),
        ],
        dim=1,
    )
    IoVtk.write_points_with_velocity(
        str(path_dir / "velocity_interpolated_at_center.vtk"),
        pix2xyz, pix2vxvyvz)

    if torch.cuda.is_available():
        RasterizedEdgeGradient.smooth_gradient(
            d_hedge2type, d_vedge2type, num_itr, d_hedge2dldr, d_vedge2dldr
        )
        print( (d_hedge2dldr.cpu()-hedge2dldr).abs().max() )
        print( (d_vedge2dldr.cpu()-vedge2dldr).abs().max() )

        d_pix2xy = pix2xy.cuda()
        d_pix2vxvy = RasterizedEdgeGradient.interpolate(
            d_hedge2dldr, d_vedge2dldr, d_pix2xy.reshape(-1, 2)
        )
        print( (d_pix2vxvy.cpu()-pix2vxvy).abs().max() )

def test_silhouette_optimization():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2occ_trg = example2()
    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    vtx2xyz.requires_grad_(True)

    lr = 10.0
    for iter in range(0, 100):
        vtx2xyz.grad = None
        bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
        pix2tri = Pix2Tri.by_raycasting(
            tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
        )
        pix2occ = (
            torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
            .to(torch.float32)
            .unsqueeze(-1)
        )
        # pix2occ = RasterizedEdgeGradient.RasterizedEdgeGradientFunction.apply(tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ)
        pix2occ = RasterizedEdgeGradient.AutogradWithSmooth.apply(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ
        )
        loss = torch.nn.functional.mse_loss(pix2occ, pix2occ_trg)
        print("iter = :", iter, "  loss=", loss.item())
        if loss.item() < 1.0e-5:
            break
        loss.backward()
        dldw_vtx2xyz = vtx2xyz.grad

        if iter == 0:
            IoVtk.write_points_with_velocity(
                str(path_dir / "silhouette_opt_ini_cpu.vtk"), vtx2xyz.detach(), dldw_vtx2xyz)

        with torch.no_grad():
            vtx2xyz -= lr * dldw_vtx2xyz

    TriMesh3.save_wavefront_obj(
        tri2vtx, vtx2xyz,
        str(path_dir / "silhouette_opt_fin_cpu.obj"))

    if torch.cuda.is_available():
        tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2occ_trg = example2()
        transform_ndc2world = transform_world2ndc.inverse().contiguous()
        transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
        transform_world2pix = transform_ndc2pix @ transform_world2ndc

        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.detach().cuda().requires_grad_(True)
        d_transform_ndc2world = transform_ndc2world.cuda()
        d_transform_world2pix = transform_world2pix.cuda()
        d_pix2occ_trg = pix2occ_trg.cuda()

        lr = 10.0
        for iter in range(0, 100):
            d_vtx2xyz.grad = None
            d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(d_tri2vtx, d_vtx2xyz)
            d_pix2tri = Pix2Tri.by_raycasting(
                d_tri2vtx, d_vtx2xyz, d_bvhnodes, d_bvhnode2aabb, d_transform_ndc2world, img_shape
            )
            d_pix2occ = (
                torch.where(d_pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
                .to(torch.float32)
                .unsqueeze(-1)
                .cuda()
            )
            # pix2occ = RasterizedEdgeGradient.RasterizedEdgeGradientFunction.apply(tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ)
            d_pix2occ = RasterizedEdgeGradient.AutogradWithSmooth.apply(
                d_tri2vtx, d_vtx2xyz, d_transform_world2pix, d_pix2tri, d_pix2occ
            )
            d_loss = torch.nn.functional.mse_loss(d_pix2occ, d_pix2occ_trg)
            print("iter = :", iter, "  loss=", d_loss.item())
            if d_loss.item() < 1.0e-5:
                break
            d_loss.backward()
            d_dldw_vtx2xyz = d_vtx2xyz.grad

            with torch.no_grad():
                d_vtx2xyz -= lr * d_dldw_vtx2xyz

        TriMesh3.save_wavefront_obj(
            d_tri2vtx.cpu(), d_vtx2xyz.cpu(),
            str(path_dir / "silhouette_opt_fin_gpu.obj"))

def example1(L_dir, L_color):
    tri2vtx0, vtx2xyz0 = TriMesh3.sphere(1.0, 64, 32)
    transform1 = Mat44.from_translation(0.0, 0, 0.0)
    vtx2xyz0 = Vtx2Xyz.transform_affine(vtx2xyz0, transform1)  # move up
    img_shape = (128, 128)

    tri2vtx, vtx2xyz = TriMesh3.sphere(1.6, 64, 32)
    transform1 = Mat44.from_translation(0.0, 0, 0.0)
    vtx2xyz = Vtx2Xyz.transform_affine(vtx2xyz, transform1)  # move up
    transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
    transform_ndc2world = transform_world2ndc.inverse()
    vtx2nrm = TriMesh3.make_vtx2normal(tri2vtx.int(), vtx2xyz)
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = Pix2Tri.by_raycasting(
        tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
    )

    from test_pix2tri import render_lambertian_shading_gouraud

    pix2rgb_target = render_lambertian_shading_gouraud(
        tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world, L_dir, L_color, pix2tri
    )
    return tri2vtx0, vtx2xyz0, transform_world2ndc, img_shape, pix2rgb_target


def test_shading_optimization():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    from test_pix2tri import generate_lighting

    light_dirs, light_colors = generate_lighting(low=0.1, mid=0.5, high=0.9)
    # Remove lighting from negative Z direction
    light_dirs = light_dirs[:-1]
    light_colors = light_colors[:-1]
    #
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2rgb_trg = example1(
        light_dirs, light_colors
    )

    img = (pix2rgb_trg.detach().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "shading_opt_trg_cpu.png")

    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    vtx2xyz.requires_grad_(True)

    from del_msh_dlpack.optimize_torch import UniformAdam

    opt = UniformAdam([vtx2xyz], lr=0.01)

    lr = 30.0
    for iter in range(0, 31):
        opt.zero_grad()
        bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
        pix2tri = Pix2Tri.by_raycasting(
            tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
        )
        vtx2nrm = TriMesh3.make_vtx2normal(tri2vtx.int(), vtx2xyz)
        from test_pix2tri import render_lambertian_shading_gouraud

        pix2rgb = render_lambertian_shading_gouraud(
            tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world, light_dirs, light_colors, pix2tri
        )
        pix2rgb = RasterizedEdgeGradient.AutogradWithSmooth.apply(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2rgb
        )
        loss = torch.nn.functional.mse_loss(pix2rgb, pix2rgb_trg)
        print("iter = :", iter, "  loss=", loss.item())
        if loss.item() < 1.0e-5:
            break
        loss.backward()
        opt.step()

        if iter % 10 == 0:
            img = (pix2rgb.detach().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(img).save(path_dir / f"shading_opt_cpu_{iter}.png")

    TriMesh3.save_wavefront_obj(tri2vtx, vtx2xyz, str(path_dir / f"shading_opt_fin.obj"))

    if torch.cuda.is_available():
        d_light_dirs, d_light_colors = generate_lighting(low=0.1, mid=0.5, high=0.9)
        d_light_dirs = d_light_dirs[:-1].cuda()
        d_light_colors = d_light_colors[:-1].cuda()
        #
        tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2rgb_trg = example1(
            light_dirs, light_colors
        )
        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.detach().cuda().requires_grad_(True)
        d_transform_world2ndc =  transform_world2ndc.cuda()
        d_pix2rgb_trg = pix2rgb_trg.cuda()
        #
        d_transform_ndc2world = transform_world2ndc.inverse().contiguous().cuda()
        d_transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape).cuda()
        d_transform_world2pix = d_transform_ndc2pix @ d_transform_world2ndc

        opt = UniformAdam([d_vtx2xyz], lr=0.01)

        for iter in range(0, 31):
            opt.zero_grad()
            d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(d_tri2vtx, d_vtx2xyz)
            d_pix2tri = Pix2Tri.by_raycasting(
                d_tri2vtx, d_vtx2xyz, d_bvhnodes, d_bvhnode2aabb, d_transform_ndc2world, img_shape
            )
            d_vtx2nrm = TriMesh3.make_vtx2normal(d_tri2vtx.int(), d_vtx2xyz)
            from test_pix2tri import render_lambertian_shading_gouraud

            d_pix2rgb = render_lambertian_shading_gouraud(
                d_tri2vtx, d_vtx2xyz, d_vtx2nrm, d_transform_ndc2world,
                d_light_dirs, d_light_colors, d_pix2tri
            )
            d_pix2rgb = RasterizedEdgeGradient.AutogradWithSmooth.apply(
                d_tri2vtx, d_vtx2xyz, d_transform_world2pix, d_pix2tri, d_pix2rgb
            )
            d_loss = torch.nn.functional.mse_loss(d_pix2rgb, d_pix2rgb_trg)
            print("iter = :", iter, "  loss=", d_loss.item())
            if d_loss.item() < 1.0e-5:
                break
            d_loss.backward()
            opt.step()

            if iter % 10 == 0:
                img = (d_pix2rgb.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
                Image.fromarray(img).save(path_dir / f"shading_opt_gpu_{iter}.png")
