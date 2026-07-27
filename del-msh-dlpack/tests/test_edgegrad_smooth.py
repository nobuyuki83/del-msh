import pathlib

from PIL import Image
import torch
import numpy as np

import del_msh_dlpack.EdgeGradSmooth.torch as EdgeGradSmooth
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz
import del_msh_dlpack.IoVtk.torch as IoVtk
import del_msh_dlpack.Vtx2Vtx.torch as Vtx2Vtx
from render_util import (
    render_lambertian_shading_gouraud,
    render_lambertian_shading_phong,
    generate_lighting,
)


def example2(resolution: int):
    tri2vtx, vtx2xyz = TriMesh3.sphere(0.5, 64, 32)
    transform1 = Mat44.from_translation(0.0, 0.0, 0.0)
    vtx2xyz = Vtx2Xyz.transform_homography(vtx2xyz, transform1)  # move up
    transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
    img_shape = (resolution, resolution)
    radius = 0.4 * resolution  # adjustable
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

    tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2occ_trg = example2(128)
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
        EdgeGradSmooth.edge_gradient_and_type(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ_src, dldw_pix2val
        )
    )

    img = ((hedge2dldr.numpy() + 0.5) * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "hedge0.png")

    img = ((vedge2dldr.numpy() + 0.5) * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "vedge0.png")

    d_hedge2type, d_vedge2type, d_hedge2dldr, d_vedge2dldr = None, None, None, None
    if torch.cuda.is_available():
        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.detach().clone().cuda()
        d_transform_world2pix = transform_world2pix.cuda()
        d_pix2tri = pix2tri.cuda()
        d_pix2occ_src = pix2occ_src.cuda()
        d_dldw_pix2val = dldw_pix2val.cuda()
        d_hedge2type, d_hedge2dldr, d_vedge2type, d_vedge2dldr = (
            EdgeGradSmooth.edge_gradient_and_type(
                d_tri2vtx,
                d_vtx2xyz,
                d_transform_world2pix,
                d_pix2tri,
                d_pix2occ_src,
                d_dldw_pix2val,
            )
        )
        assert torch.equal(d_hedge2type.cpu(), hedge2type)
        assert (d_hedge2dldr.cpu() - hedge2dldr).abs().max() < 1.0e-8
        assert torch.equal(d_vedge2type.cpu(), vedge2type)
        assert (d_vedge2dldr.cpu() - vedge2dldr).abs().max() < 1.0e-8

    num_itr = 10000
    EdgeGradSmooth.smooth_gradient(
        hedge2type,
        vedge2type,
        num_itr,
        hedge2dldr,
        vedge2dldr,
    )
    img = ((hedge2dldr.numpy() + 0.5) * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "hedge1.png")

    img = ((vedge2dldr.numpy() + 0.5) * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "vedge1.png")

    IoVtk.write_velocity_on_staggered_grid(
        str(path_dir / "velocity_on_staggered_grid.vtk"), hedge2dldr, vedge2dldr
    )

    # ------------------------
    # test interpolate

    img_h, img_w = img_shape[1], img_shape[0]
    ys = torch.arange(img_h, dtype=torch.float32) + 0.5  # (img_h,)
    xs = torch.arange(img_w, dtype=torch.float32) + 0.5  # (img_w,)
    pix2xy = torch.stack(
        torch.meshgrid(xs, ys, indexing="xy"), dim=-1
    )  # (img_h, img_w, 2): [..., 0]=x, [..., 1]=y
    pix2vxvy = EdgeGradSmooth.interpolate(hedge2dldr, vedge2dldr, pix2xy.reshape(-1, 2))
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
        str(path_dir / "velocity_interpolated_at_center.vtk"), pix2xyz, pix2vxvyvz
    )

    if torch.cuda.is_available():
        EdgeGradSmooth.smooth_gradient(
            d_hedge2type, d_vedge2type, num_itr, d_hedge2dldr, d_vedge2dldr
        )
        print((d_hedge2dldr.cpu() - hedge2dldr).abs().max())
        print((d_vedge2dldr.cpu() - vedge2dldr).abs().max())

        d_pix2xy = pix2xy.cuda()
        d_pix2vxvy = EdgeGradSmooth.interpolate(
            d_hedge2dldr, d_vedge2dldr, d_pix2xy.reshape(-1, 2)
        )
        print((d_pix2vxvy.cpu() - pix2vxvy).abs().max())


def test_silhouette_optimization():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    nres = 256
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2occ_trg = example2(nres)
    vtx2vtx = Vtx2Vtx.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)
    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
    transform_world2pix = transform_ndc2pix @ transform_world2ndc
    #
    vtx2xyz.requires_grad_(True)

    num_screen_smooth = 300
    num_mesh_smooth = 3

    from del_msh_dlpack.optimize_torch import UniformAdam

    opt = UniformAdam([vtx2xyz], lr=0.01)

    for iter in range(0, 100):
        # vtx2xyz.grad = None
        opt.zero_grad()
        bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
        pix2tri = Pix2Tri.by_raycasting(
            tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
        )
        pix2occ = (
            torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
            .to(torch.float32)
            .unsqueeze(-1)
        )
        pix2occ = EdgeGradSmooth.Autograd.apply(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ, num_screen_smooth
        )
        loss = torch.nn.functional.mse_loss(pix2occ, pix2occ_trg)
        print("iter = :", iter, "  loss=", loss.item())
        if loss.item() < 1.0e-5:
            break
        loss.backward()

        with torch.no_grad():
            dldw_vtx2xyz = vtx2xyz.grad.detach().clone()
            dldw_vtx2xyz0 = dldw_vtx2xyz.detach().clone()
            Vtx2Vtx.laplacian_smoothing(
                vtx2vtx[0],
                vtx2vtx[1],
                0.0,
                dldw_vtx2xyz0,
                dldw_vtx2xyz,
                num_mesh_smooth,
            )
            vtx2xyz.grad[:, :] = dldw_vtx2xyz[:, :]

        if iter == 0:
            IoVtk.write_points_with_velocity(
                str(path_dir / "silhouette_opt_ini_cpu.vtk"),
                vtx2xyz.detach(),
                dldw_vtx2xyz,
            )
        opt.step()

    TriMesh3.save_wavefront_obj(
        tri2vtx, vtx2xyz, str(path_dir / "silhouette_opt_fin_cpu.obj")
    )

    if torch.cuda.is_available():
        tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2occ_trg = example2(nres)
        transform_ndc2world = transform_world2ndc.inverse().contiguous()
        transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape)
        transform_world2pix = transform_ndc2pix @ transform_world2ndc

        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.detach().cuda().requires_grad_(True)
        d_transform_ndc2world = transform_ndc2world.cuda()
        d_transform_world2pix = transform_world2pix.cuda()
        d_pix2occ_trg = pix2occ_trg.cuda()

        opt = UniformAdam([d_vtx2xyz], lr=0.01)

        lr = 10.0
        for iter in range(0, 100):
            d_vtx2xyz.grad = None
            opt.zero_grad()
            d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(
                d_tri2vtx, d_vtx2xyz
            )
            d_pix2tri = Pix2Tri.by_raycasting(
                d_tri2vtx,
                d_vtx2xyz,
                d_bvhnodes,
                d_bvhnode2aabb,
                d_transform_ndc2world,
                img_shape,
            )
            d_pix2occ = (
                torch.where(d_pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
                .to(torch.float32)
                .unsqueeze(-1)
                .cuda()
            )
            d_vtx2vtx = (vtx2vtx[0].cuda(), vtx2vtx[1].cuda())
            # pix2occ = EdgeGrad.RasterizedEdgeGradientFunction.apply(tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ)
            d_pix2occ = EdgeGradSmooth.Autograd.apply(
                d_tri2vtx,
                d_vtx2xyz,
                d_transform_world2pix,
                d_pix2tri,
                d_pix2occ,
                num_screen_smooth,
            )
            d_loss = torch.nn.functional.mse_loss(d_pix2occ, d_pix2occ_trg)
            print("iter = :", iter, "  loss=", d_loss.item())
            if d_loss.item() < 1.0e-5:
                break
            d_loss.backward()
            d_dldw_vtx2xyz = d_vtx2xyz.grad
            """
            with torch.no_grad():
                d_vtx2xyz -= lr * d_dldw_vtx2xyz
            """
            opt.step()

        TriMesh3.save_wavefront_obj(
            d_tri2vtx.cpu(),
            d_vtx2xyz.cpu(),
            str(path_dir / "silhouette_opt_fin_gpu.obj"),
        )


def example1(L_dir, L_color):
    tri2vtx0, vtx2xyz0 = TriMesh3.sphere(1.0, 64, 32)
    transform1 = Mat44.from_translation(0.0, 0, 0.0)
    vtx2xyz0 = Vtx2Xyz.transform_homography(vtx2xyz0, transform1)  # move up
    img_shape = (128, 128)

    tri2vtx, vtx2xyz = TriMesh3.sphere(1.6, 64, 32)
    transform1 = Mat44.from_translation(0.0, 0, 0.0)
    vtx2xyz = Vtx2Xyz.transform_homography(vtx2xyz, transform1)  # move up
    transform_world2ndc = Mat44.from_scale(0.5, 0.5, 0.5)
    transform_ndc2world = transform_world2ndc.inverse()
    vtx2nrm = TriMesh3.make_vtx2normal(tri2vtx.int(), vtx2xyz)
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = Pix2Tri.by_raycasting(
        tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
    )

    pix2rgb_target = render_lambertian_shading_gouraud(
        tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world, L_dir, L_color, pix2tri
    )
    return tri2vtx0, vtx2xyz0, transform_world2ndc, img_shape, pix2rgb_target


def render_directional_silhouette(
    tri2vtx,
    vtx2xyz,
    view2world2ndc,
    img_shape,
    *,
    num_screen_smoothing=100,
):
    """Render geometry with directional lighting using del-msh + EdgeGrad.

    Args:
        vtx2xyz: Vertex positions (N, 3).
        tri2vtx: Triangle indices (M, 3), int32.
        view2world2ndc: Camera matrices (B, 4, 4) (world-to-NDC per view).
        light2dir: Lighting directions (K, 3).
        light2color: Lighting colors (K, 3).
        resolution: Tuple (height, width) for rasterization.
        background: Background color tensor (3,).
        smooth_edge_gradient: If true, use del-msh's 100-iteration
            screen-space gradient smoothing.  If false, use its raw
            rasterized edge gradient.

    Returns:
        Rendered color image tensor with shape (B, H, W, 3).
    """
    device = vtx2xyz.device

    # del-msh expects uint32 connectivity; keep the int32 `tri` for torch indexing.
    tri2vtx = tri2vtx.to(torch.uint32).contiguous()

    # Acceleration structure (BVH) for ray-casting. Unlike the antialias back-end
    # we do NOT need the edge2vtx / edge2tri silhouette connectivity here --
    # EdgeGrad works directly on the rasterized pixel grid.
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)

    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape, device=device)

    uint32_max = torch.iinfo(torch.uint32).max

    images = []
    # greencloud uses OpenGL NDC (near=-1, far=+1), whereas del-msh's ray
    # generator starts at z=+1 and marches toward z=-1.  Flip NDC z before
    # unprojection so ray casting selects the camera-facing surface.
    ndc_z_flip = torch.eye(4, dtype=vtx2xyz.dtype, device=device)
    ndc_z_flip[2, 2] = -1
    for i in range(view2world2ndc.shape[0]):
        transform_world2ndc = view2world2ndc[i]
        transform_ndc2world = (transform_world2ndc.inverse() @ ndc_z_flip).contiguous()
        transform_world2pix = transform_ndc2pix @ transform_world2ndc

        # Rasterize: triangle index per pixel (non-differentiable visibility).
        pix2tri = Pix2Tri.by_raycasting(
            tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
        )

        pix2occ = (
            torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
            .to(torch.float32)
            .unsqueeze(-1)
            .cuda()
        )

        # Edge-gradient estimator (del-msh's analogue of dr.antialias / DRTK's
        # edge_grad): leaves the forward image unchanged but backpropagates
        # silhouette gradients w.r.t. vertex positions. Handles all channels at
        # once, so no per-channel loop is needed.
        """
        edge_gradient = (
            EdgeGrad.AutogradWithSmooth
            if smooth_edge_gradient
            else EdgeGrad.RasterizedEdgeGradientFunction
        )
        """
        color = EdgeGradSmooth.Autograd.apply(
            tri2vtx,
            vtx2xyz,
            transform_world2pix,
            pix2tri,
            pix2occ.contiguous(),
            num_screen_smoothing,
        )

        images.append(color)

    return torch.stack(images, dim=0)


def test_silhouette_opt_multiview():
    if not torch.cuda.is_available():
        return
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)

    tri2vtx0, vtx2xyz0 = TriMesh3.sphere(1.0, 64, 32)
    tri2vtx0 = tri2vtx0.cuda()
    vtx2xyz0 = vtx2xyz0.cuda()
    vtx2vtx0 = Vtx2Vtx.from_uniform_mesh(tri2vtx0, vtx2xyz0.shape[0], False)
    """
    aabb = torch.stack([vtx2xyz0.amin(dim=0), vtx2xyz0.amax(dim=0)], dim=0)
    distance = (aabb[1] - aabb[0]).norm() * 1.4
    target = aabb.mean(dim=0, keepdim=True).float()

    view2origin = fibonacci_camera_origin(target, 1, distance)

    from del_msh_dlpack.PerspectiveCamera.torch import PerspectiveCamera

    view2world2ndc = (
        PerspectiveCamera(
            near=torch.tensor([1e-2]).float(),
            far=torch.tensor([distance * 3.0]).float(),
        )
        .look_at(
            origin=view2origin,
            target=target,
            up=torch.tensor([[0, 0, 1]]).float(),
        )
        .cuda()
    )
    """
    view2world2ndc = Mat44.from_scale(0.5, 0.5, 0.5).unsqueeze(0).cuda()
    print(view2world2ndc.shape)
    #
    img_shape = (128, 128)
    #
    view2pix2occ0 = render_directional_silhouette(
        tri2vtx0,
        vtx2xyz0,
        view2world2ndc,
        img_shape,
    )

    for i in range(view2pix2occ0.shape[0]):
        filename = path_dir / f"reference_{i:05d}.png"
        img = (
            (view2pix2occ0[i].cpu().squeeze().numpy() * 255)
            .clip(0, 255)
            .astype("uint8")
        )
        Image.fromarray(img).save(filename)

    tri2vtx, vtx2xyz = TriMesh3.sphere(0.1, 64, 32)
    tri2vtx = tri2vtx.cuda()
    vtx2xyz = vtx2xyz.cuda().requires_grad_(True)
    vtx2vtx = Vtx2Vtx.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)

    from del_msh_dlpack.optimize_torch import UniformAdam

    opt = UniformAdam([vtx2xyz], lr=0.01)

    for i_iter in range(301):
        torch.cuda.synchronize()
        view2pix2occ = render_directional_silhouette(
            tri2vtx,
            vtx2xyz,
            view2world2ndc,
            img_shape,
            num_screen_smoothing=100,
        )
        # loss = image_pyramid_loss(reference, color, num_scales=4)
        loss = torch.nn.functional.mse_loss(view2pix2occ, view2pix2occ0)
        print(i_iter, loss.item())
        loss.backward()

        with torch.no_grad():
            dldw_vtx2xyz = vtx2xyz.grad.detach().clone()
            dldw_vtx2xyz0 = dldw_vtx2xyz.clone()
            Vtx2Vtx.laplacian_smoothing(
                vtx2vtx[0], vtx2vtx[1], 0.0, dldw_vtx2xyz, dldw_vtx2xyz0, 10
            )
            vtx2xyz.grad[:, :] = dldw_vtx2xyz0[:, :]
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()

        # optimization_times.append(time.perf_counter() - iteration_start)
        # remeshing_times.append(remeshing_time)
        # losses.append(loss.item())
        # mae.append((reference - color.detach()).abs().mean().item())

        if i_iter % 50 == 0:
            for i in range(view2pix2occ.shape[0]):
                filename = path_dir / f"reference_{i_iter}_{i:05d}.png"
                img = (
                    (view2pix2occ[i].detach().cpu().squeeze().numpy() * 255)
                    .clip(0, 255)
                    .astype("uint8")
                )
                Image.fromarray(img).save(filename)

    TriMesh3.save_wavefront_obj(
        tri2vtx.cpu(), vtx2xyz.cpu(), str(path_dir / "final.obj")
    )


def render_directional(
    tri2vtx,
    vtx2vtx,
    vtx2xyz,
    view2world2ndc,
    light2dir,
    light2color,
    img_shape,
    background,
    *,
    num_screen_smoothing=100,
    num_mesh_smoothing=1,
):
    """Render geometry with directional lighting using del-msh + EdgeGrad.

    Args:
        vtx2xyz: Vertex positions (N, 3).
        tri2vtx: Triangle indices (M, 3), int32.
        view2world2ndc: Camera matrices (B, 4, 4) (world-to-NDC per view).
        light2dir: Lighting directions (K, 3).
        light2color: Lighting colors (K, 3).
        resolution: Tuple (height, width) for rasterization.
        background: Background color tensor (3,).
        smooth_edge_gradient: If true, use del-msh's 100-iteration
            screen-space gradient smoothing.  If false, use its raw
            rasterized edge gradient.

    Returns:
        Rendered color image tensor with shape (B, H, W, 3).
    """
    device = vtx2xyz.device

    # del-msh expects uint32 connectivity; keep the int32 `tri` for torch indexing.
    tri2vtx = tri2vtx.to(torch.uint32).contiguous()

    # Per-vertex directional shading (differentiable w.r.t. pos through the normals).
    nrm = TriMesh3.make_vtx2normal(tri2vtx.int(), vtx2xyz)
    shading = torch.zeros_like(nrm[..., :3])
    for i in range(len(light2dir)):
        dir = torch.nn.functional.normalize(light2dir[i], dim=0)
        ndotl = torch.clamp((nrm @ dir).unsqueeze(-1), min=0.0)
        shading += ndotl * light2color[i]

    # Acceleration structure (BVH) for ray-casting. Unlike the antialias back-end
    # we do NOT need the edge2vtx / edge2tri silhouette connectivity here --
    # EdgeGrad works directly on the rasterized pixel grid.
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)

    transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape, device=device)

    uint32_max = torch.iinfo(torch.uint32).max

    images = []
    # greencloud uses OpenGL NDC (near=-1, far=+1), whereas del-msh's ray
    # generator starts at z=+1 and marches toward z=-1.  Flip NDC z before
    # unprojection so ray casting selects the camera-facing surface.
    ndc_z_flip = torch.eye(4, dtype=vtx2xyz.dtype, device=device)
    ndc_z_flip[2, 2] = -1
    for i in range(view2world2ndc.shape[0]):
        transform_world2ndc = view2world2ndc[i]
        transform_ndc2world = (transform_world2ndc.inverse() @ ndc_z_flip).contiguous()
        transform_world2pix = transform_ndc2pix @ transform_world2ndc

        # Rasterize: triangle index per pixel (non-differentiable visibility).
        pix2tri = Pix2Tri.by_raycasting(
            tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
        )

        # Perspective-correct interpolation of the per-vertex shading (differentiable).
        color = Pix2Tri.interpolate(
            pix2tri, tri2vtx, vtx2xyz, shading, transform_ndc2world
        )

        alpha = (pix2tri != uint32_max).unsqueeze(-1).to(torch.float32)
        color = color * alpha + background * (1 - alpha)

        # Edge-gradient estimator (del-msh's analogue of dr.antialias / DRTK's
        # edge_grad): leaves the forward image unchanged but backpropagates
        # silhouette gradients w.r.t. vertex positions. Handles all channels at
        # once, so no per-channel loop is needed.
        """
        edge_gradient = (
            EdgeGrad.AutogradWithSmooth
            if smooth_edge_gradient
            else EdgeGrad.RasterizedEdgeGradientFunction
        )
        """
        color = EdgeGradSmooth.Autograd.apply(
            tri2vtx,
            vtx2xyz,
            transform_world2pix,
            pix2tri,
            color.contiguous(),
            num_screen_smoothing,
        )

        images.append(color)

    return torch.stack(images, dim=0)


def save_image(
    img_tensor: torch.Tensor,
    filename: str | pathlib.Path,
    gamma: float = 2.2,
    flip_vertical: bool = False,
) -> None:
    """Save a single image tensor with gamma correction.

    Args:
        img_tensor:
            Image tensor with shape (H, W, 3) and values in [0, 1].
        filename:
            Output image filename.
        gamma:
            Gamma value for correction.
        flip_vertical:
            Flip rows before saving. This is needed for backends whose
            framebuffer origin is bottom-left, such as nvdiffrast.
    """
    if img_tensor.ndim != 3 or img_tensor.shape[-1] != 3:
        raise ValueError(
            f"Expected an image tensor with shape (H, W, 3), "
            f"but got {tuple(img_tensor.shape)}"
        )

    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, but got {gamma}")

    img = img_tensor.detach().cpu().numpy()

    img = np.clip(img, 0.0, 1.0)
    img = img ** (1.0 / gamma)
    img = np.round(img * 255.0).astype(np.uint8)

    if flip_vertical:
        img = np.flip(img, axis=0)

    # np.flip() may produce an array with negative strides.
    img = np.ascontiguousarray(img)

    Image.fromarray(img, mode="RGB").save(filename)


def test_shading_optimization():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    light2dir, light2color = generate_lighting(low=0.1, mid=0.5, high=0.9)
    # Remove lighting from negative Z direction
    light2dir = light2dir[:-1]
    light2color = light2color[:-1]
    #
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2rgb_trg = example1(
        light2dir, light2color
    )
    vtx2vtx = Vtx2Vtx.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)

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

        pix2rgb = render_lambertian_shading_gouraud(
            tri2vtx,
            vtx2xyz,
            vtx2nrm,
            transform_ndc2world,
            light2dir,
            light2color,
            pix2tri,
        )
        pix2rgb = EdgeGradSmooth.Autograd.apply(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2rgb, 100
        )
        loss = torch.nn.functional.mse_loss(pix2rgb, pix2rgb_trg)
        print("iter = :", iter, "  loss=", loss.item())
        if loss.item() < 1.0e-5:
            break
        loss.backward()
        with torch.no_grad():
            dldw_vtx2xyz = vtx2xyz.grad.detach().clone()
            dldw_vtx2xyz0 = dldw_vtx2xyz.detach().clone()
            Vtx2Vtx.laplacian_smoothing(
                vtx2vtx[0],
                vtx2vtx[1],
                0.0,
                dldw_vtx2xyz0,
                dldw_vtx2xyz,
                10,
            )
            vtx2xyz.grad[:, :] = dldw_vtx2xyz[:, :]
        opt.step()

        if iter % 10 == 0:
            img = (pix2rgb.detach().numpy() * 255).clip(0, 255).astype("uint8")
            Image.fromarray(img).save(path_dir / f"shading_opt_cpu_{iter}.png")

    TriMesh3.save_wavefront_obj(
        tri2vtx, vtx2xyz, str(path_dir / f"shading_opt_fin.obj")
    )

    if torch.cuda.is_available():
        d_light_dirs, d_light_colors = generate_lighting(low=0.1, mid=0.5, high=0.9)
        d_light_dirs = d_light_dirs[:-1].cuda()
        d_light_colors = d_light_colors[:-1].cuda()
        #
        tri2vtx, vtx2xyz, transform_world2ndc, img_shape, pix2rgb_trg = example1(
            light2dir, light2color
        )
        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.detach().cuda().requires_grad_(True)
        d_transform_world2ndc = transform_world2ndc.cuda()
        d_pix2rgb_trg = pix2rgb_trg.cuda()
        #
        d_transform_ndc2world = transform_world2ndc.inverse().contiguous().cuda()
        d_transform_ndc2pix = Mat44.from_transform_ndc2pix(img_shape).cuda()
        d_transform_world2pix = d_transform_ndc2pix @ d_transform_world2ndc

        opt = UniformAdam([d_vtx2xyz], lr=0.01)

        for iter in range(0, 31):
            opt.zero_grad()
            d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(
                d_tri2vtx, d_vtx2xyz
            )
            d_pix2tri = Pix2Tri.by_raycasting(
                d_tri2vtx,
                d_vtx2xyz,
                d_bvhnodes,
                d_bvhnode2aabb,
                d_transform_ndc2world,
                img_shape,
            )
            d_vtx2nrm = TriMesh3.make_vtx2normal(d_tri2vtx.int(), d_vtx2xyz)

            d_pix2rgb = render_lambertian_shading_gouraud(
                d_tri2vtx,
                d_vtx2xyz,
                d_vtx2nrm,
                d_transform_ndc2world,
                d_light_dirs,
                d_light_colors,
                d_pix2tri,
            )
            d_pix2rgb = EdgeGradSmooth.Autograd.apply(
                d_tri2vtx, d_vtx2xyz, d_transform_world2pix, d_pix2tri, d_pix2rgb, 100
            )
            d_loss = torch.nn.functional.mse_loss(d_pix2rgb, d_pix2rgb_trg)
            print("iter = :", iter, "  loss=", d_loss.item())
            if d_loss.item() < 1.0e-5:
                break
            d_loss.backward()
            opt.step()

            if iter % 10 == 0:
                img = (
                    (d_pix2rgb.detach().cpu().numpy() * 255)
                    .clip(0, 255)
                    .astype("uint8")
                )
                Image.fromarray(img).save(path_dir / f"shading_opt_gpu_{iter}.png")


def test_shading_opt_multiview():
    if not torch.cuda.is_available():
        return
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)

    tri2vtx0, vtx2xyz0 = TriMesh3.sphere(1.0, 64, 32)
    tri2vtx0 = tri2vtx0.cuda()
    vtx2xyz0 = vtx2xyz0.cuda()
    vtx2vtx0 = Vtx2Vtx.from_uniform_mesh(tri2vtx0, vtx2xyz0.shape[0], False)
    aabb = torch.stack([vtx2xyz0.amin(dim=0), vtx2xyz0.amax(dim=0)], dim=0)
    distance = (aabb[1] - aabb[0]).norm() * 1.4
    target = aabb.mean(dim=0, keepdim=True).float()

    import del_msh_dlpack.PerspectiveCamera.torch as PerspectiveCamera

    view2origin = PerspectiveCamera.fibonacci_camera_origin(target, 4, distance)

    view2world2ndc = (
        PerspectiveCamera.PerspectiveCamera(
            near=torch.tensor([1e-2]).float(),
            far=torch.tensor([distance * 10.0]).float(),
        )
        .look_at(
            origin=view2origin,
            target=target,
            up=torch.tensor([[0, 0, 1]]).float(),
        )
        .cuda()
    )
    #
    light2dir, light2color = generate_lighting(low=0.1, mid=0.5, high=0.9)
    # Remove lighting from negative Z direction
    light2dir = light2dir[:-1].cuda()
    light2color = light2color[:-1].cuda()
    #
    background = torch.zeros(3, device=vtx2xyz0.device)
    img_shape = (128, 128)
    #
    view2pix2rgb0 = render_directional(
        tri2vtx0,
        vtx2vtx0,
        vtx2xyz0,
        view2world2ndc,
        light2dir,
        light2color,
        img_shape,
        background,
    )

    for i in range(view2pix2rgb0.shape[0]):
        filename = path_dir / f"reference_{i:05d}.png"
        save_image(view2pix2rgb0[i], filename)

    tri2vtx, vtx2xyz = TriMesh3.sphere(0.8, 64, 32)
    tri2vtx = tri2vtx.cuda()
    vtx2xyz = vtx2xyz.cuda().requires_grad_(True)
    vtx2vtx = Vtx2Vtx.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)

    from del_msh_dlpack.optimize_torch import UniformAdam

    opt = UniformAdam([vtx2xyz], lr=0.01)

    lr = 10.0

    for i_iter in range(301):
        torch.cuda.synchronize()
        vtx2xyz.grad = None
        opt.zero_grad()
        view2pix2rgb = render_directional(
            tri2vtx,
            vtx2vtx,
            vtx2xyz,
            view2world2ndc,
            light2dir,
            light2color,
            img_shape,
            background,
            num_screen_smoothing=100,
            num_mesh_smoothing=1,
        )
        # loss = image_pyramid_loss(reference, color, num_scales=4)
        loss = torch.nn.functional.mse_loss(view2pix2rgb, view2pix2rgb0)
        print(i_iter, loss.item())
        loss.backward()

        with torch.no_grad():
            dldw_vtx2xyz = vtx2xyz.grad.detach().clone()
            dldw_vtx2xyz0 = dldw_vtx2xyz.detach().clone()
            Vtx2Vtx.laplacian_smoothing(
                vtx2vtx[0], vtx2vtx[1], 0.0, dldw_vtx2xyz0, dldw_vtx2xyz, 3
            )
            vtx2xyz.grad = dldw_vtx2xyz

        torch.cuda.synchronize()
        opt.step()

        """
        dldw_vtx2xyz = vtx2xyz.grad
        with torch.no_grad():
            vtx2xyz -= lr * dldw_vtx2xyz
        """

        opt.zero_grad()

        # optimization_times.append(time.perf_counter() - iteration_start)
        # remeshing_times.append(remeshing_time)
        # losses.append(loss.item())
        # mae.append((reference - color.detach()).abs().mean().item())

        if i_iter % 50 == 0:
            for i in range(view2pix2rgb.shape[0]):
                filename = path_dir / f"reference_{i_iter}_{i:05d}.png"
                save_image(view2pix2rgb[i], filename)

    TriMesh3.save_wavefront_obj(
        tri2vtx.cpu(), vtx2xyz.cpu(), str(path_dir / "final.obj")
    )
