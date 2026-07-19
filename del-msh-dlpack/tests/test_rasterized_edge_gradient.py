import pathlib

#
from PIL import Image
import torch

#
import del_msh_dlpack.RasterizedEdgeGradient.torch as RasterizedEdgeGradient
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz
import del_msh_dlpack.IoVtk.torch as IoVtk
import test_pix2tri


#


def test_gradient_visualization_silhouette():
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

    # gradient visualization: d(pixel) / d(vtx) projected onto x-direction
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
        dldw_vtx2xyz = RasterizedEdgeGradient.gradient(
            tri2vtx, vtx2xyz, transform_world2pix, dldw_pix2occ, pix2tri
        )
        dpix = (dxyz * dldw_vtx2xyz).sum().item()
        c = apply_colormap_bwr(dpix, vmin, vmax)
        pix2rgb_diff.view(-1, 3)[i_pix] = torch.tensor(c, dtype=torch.uint8)
    #
    path1 = (
        pathlib.Path(__file__).parent.parent.parent
        / "target"
        / "del_msh_dlpack__microedge6.png"
    )
    Image.fromarray(pix2rgb_diff.numpy()).save(path1)
    #


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
    path0 = (
        pathlib.Path(__file__).parent.parent.parent
        / "target"
        / "del_msh_dlpack__microedge0.png"
    )
    path0.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path0)
    #
    img = (pix2occ_trg.squeeze().numpy() * 255).clip(0, 255).astype("uint8")
    path0 = (
        pathlib.Path(__file__).parent.parent.parent
        / "target"
        / "del_msh_dlpack__microedge1.png"
    )
    path0.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path0)
    #
    dldw_pix2val = pix2occ_src - pix2occ_trg
    hedge2type, hedge2dldr, vedge2type, vedge2dldr = (
        RasterizedEdgeGradient.edge_gradient_and_type(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ_src, dldw_pix2val
        )
    )

    # torch.set_printoptions(edgeitems=130)
    # print(hedge2type)

    RasterizedEdgeGradient.smooth_gradient(
        hedge2type, hedge2dldr, vedge2type, vedge2dldr
    )

    path0 = (
        pathlib.Path(__file__).parent.parent.parent
        / "target"
        / "del_msh_dlpack__microedge2.vtk"
    )
    IoVtk.write_velocity_on_staggered_grid(path0, hedge2dldr, vedge2dldr)

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

    path0 = (
        pathlib.Path(__file__).parent.parent.parent
        / "target"
        / "del_msh_dlpack__microedge3.vtk"
    )
    IoVtk.write_points_with_velocity(str(path0), pix2xyz, pix2vxvyvz)


def test_silhouette_optimization():
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
        # print(pix2occ.shape)
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
            path0 = (
                pathlib.Path(__file__).parent.parent.parent
                / "target"
                / "del_msh_dlpack__microedge4.vtk"
            )
            IoVtk.write_points_with_velocity(str(path0), vtx2xyz.detach(), dldw_vtx2xyz)

        """
        img = (pix2occ.squeeze().detach().numpy() * 255).clip(0, 255).astype('uint8')
        path0 = pathlib.Path(__file__).parent.parent.parent / "target" / f"del_msh_dlpack__microedge8_{iter}.png"
        path0.parent.mkdir(exist_ok=True)
        Image.fromarray(img).save(path0)
        """

        with torch.no_grad():
            vtx2xyz -= lr * dldw_vtx2xyz

        path0 = (
            pathlib.Path(__file__).parent.parent.parent
            / "target"
            / "del_msh_dlpack__microedge5.obj"
        )
        TriMesh3.save_wavefront_obj(tri2vtx, vtx2xyz, str(path0))


def example1(L_dir, L_color):
    tri2vtx0, vtx2xyz0 = TriMesh3.sphere(1.0, 64, 32)
    transform1 = Mat44.from_translation(0.0, 0, 0.0)
    vtx2xyz = Vtx2Xyz.transform_affine(vtx2xyz0, transform1)  # move up
    transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
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
    from test_pix2tri import generate_lighting

    L_dirs, L_colors = generate_lighting(low=0.1, mid=0.5, high=0.9, device="cpu")
    # Remove lighting from negative Z direction
    L_dirs = L_dirs[:-1]
    L_colors = L_colors[:-1]

    tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2rgb_trg = example1(
        L_dirs, L_colors
    )

    img = (pix2rgb_trg.detach().numpy() * 255).clip(0, 255).astype("uint8")
    path0 = (
        pathlib.Path(__file__).parent.parent.parent
        / "target"
        / "del_msh_dlpack__microedge6.png"
    )
    path0.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path0)

    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    vtx2xyz.requires_grad_(True)

    from del_msh_dlpack.optimize_torch import UniformAdam

    opt = UniformAdam([vtx2xyz], lr=0.01)

    lr = 30.0
    for iter in range(0, 30):
        opt.zero_grad()
        bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
        pix2tri = Pix2Tri.by_raycasting(
            tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
        )
        vtx2nrm = TriMesh3.make_vtx2normal(tri2vtx.int(), vtx2xyz)
        # from test_pix2tri import render_lambertian_shading_phong
        from test_pix2tri import render_lambertian_shading_gouraud

        pix2rgb = render_lambertian_shading_gouraud(
            tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world, L_dirs, L_colors, pix2tri
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

        # dldw_vtx2xyz = vtx2xyz.grad
        """
        if iter == 0:
            path0 = pathlib.Path(__file__).parent.parent.parent / "target" / "del_msh_dlpack__microedge7.vtk"
            IoVtk.write_points_with_velocity(str(path0), vtx2xyz.detach(), dldw_vtx2xyz)
        """
        img = (pix2rgb.detach().numpy() * 255).clip(0, 255).astype("uint8")
        path0 = (
            pathlib.Path(__file__).parent.parent.parent
            / "target"
            / f"del_msh_dlpack__microedge8_{iter}.png"
        )
        path0.parent.mkdir(exist_ok=True)
        Image.fromarray(img).save(path0)

        """
        with torch.no_grad():
            vtx2xyz -= lr * dldw_vtx2xyz
        """

        path0 = (
            pathlib.Path(__file__).parent.parent.parent
            / "target"
            / "del_msh_dlpack__microedge9.obj"
        )
        TriMesh3.save_wavefront_obj(tri2vtx, vtx2xyz, str(path0))
