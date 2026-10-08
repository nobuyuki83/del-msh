import pathlib, time, csv

from PIL import Image
import torch

import del_msh_dlpack.EdgeGrad.torch as RasterizedEdgeGradient
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz


def test_gradient_visualization_silhouette():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "out_dlpack"
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
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "out_dlpack"
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


def render(image_shape, views, tri2vtx, vtx2xyz):
    device = tri2vtx.device
    import del_msh_dlpack.Pix2Depth.torch as Pix2Depth
    import del_msh_dlpack.EdgeGrad.torch as EdgeGrad

    # 各ビューの深度画像をまとめるため、アトラス全体のバッファを用意する。
    pix2depth = torch.zeros((image_shape[1], image_shape[0]), device=device)
    pix2occ = torch.zeros((image_shape[1], image_shape[0]), device=device)

    for i_view, view in enumerate(views):
        transform_world2ndc, viewport = view
        transform_world2ndc = transform_world2ndc.to(device)

        # ラスタライズ側はこのビューの逆変換行列を受け取る。
        transform_ndc2world = torch.inverse(transform_world2ndc).contiguous()
        bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)

        transform_ndc2pix = Mat44.from_transform_ndc2pix(
            (viewport[2], viewport[3]), device=device
        )
        transform_world2pix = transform_ndc2pix @ transform_world2ndc

        # まず各ピクセルで見えている三角形を求め、その後に深度を計算する。
        pix2tri = Pix2Tri.by_raycasting(
            tri2vtx,
            vtx2xyz,
            bvhnodes,
            bvhnode2aabb,
            transform_ndc2world,
            (viewport[2], viewport[3]),
        )

        pix2depth_view = Pix2Depth.AutogradFunction.apply(
            vtx2xyz, pix2tri, tri2vtx, transform_ndc2world
        )

        pix2occ_view = (
            torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0)
            .to(torch.float32)
            .to(device)
        )
        pix2occ_view = EdgeGrad.Autograd.apply(
            tri2vtx, vtx2xyz, transform_world2pix, pix2tri, pix2occ_view.unsqueeze(-1)
        ).squeeze(-1)

        """
        pix2occ = torch.where(pix2tri == torch.iinfo(torch.uint32).max, 0.0, 1.0).to(torch.float32)
        img = (pix2occ.numpy() * 255).clip(0, 255).astype('uint8')
        path0 = path_dir_trg / f"test_edgegrad_silhouette_{i_view}.png"
        Image.fromarray(img).save(path0)
        """

        # ビューごとの深度タイルをアトラス内の対応領域へ書き込む。
        pix2depth[
            viewport[1] : viewport[1] + viewport[3],
            viewport[0] : viewport[0] + viewport[2],
        ] = pix2depth_view
        pix2occ[
            viewport[1] : viewport[1] + viewport[3],
            viewport[0] : viewport[0] + viewport[2],
        ] = pix2occ_view

    return pix2depth, pix2occ


def make_problem():
    path_dir_asset = pathlib.Path(__file__).parent.parent.parent / "asset"
    import del_msh_dlpack.IoOff.torch as IoOff

    tri2vtx, vtx2xyz = IoOff.load_tri_mesh(str(path_dir_asset / "propeller1.off"))

    # ビュー生成側ではメッシュの AABB だけを使う。
    aabb = torch.stack([vtx2xyz.min(axis=0).values, vtx2xyz.max(axis=0).values])

    # 6 方向の正投影ビューと、その配置先アトラスを計算する。
    from del_msh_dlpack import util_views

    image_shape, views = util_views.make_views(aabb, 100, 0.9)

    pix2depth_trg, pix2occ_trg = render(image_shape, views, tri2vtx, vtx2xyz)

    return pix2depth_trg, pix2occ_trg, views


def sample_aabb(aabb_min, aabb_max, num_sample):
    """
    Uniformly sample random 3D points inside an axis-aligned bounding box.

    Args:
        aabb_min: (3,) tensor of the minimum corner of the bounding box.
        aabb_max: (3,) tensor of the maximum corner of the bounding box.
        num_sample: Number of points to sample.

    Returns:
        samples: (num_sample, 3) tensor of sampled points.
    """
    u = torch.rand((num_sample, 3), dtype=torch.float32, device=aabb_min.device)
    samples = aabb_min + (aabb_max - aabb_min) * u
    return samples

def fit(device: torch.device, tri2vtx, vtx2xyz, wtx2xyz, pix2depth_trg, pix2occ_trg, views):
    from del_msh_dlpack.NBody import Elastic
    #
    lr = 0.1
    num_substep = 2  # Number of sub-steps for the Green's function filter per iteration
    num_itr = 251  # Total number of optimization iterations
    num_interval_save_file = 250  # Save intermediate results every this many iterations
    model_filter_body = Elastic(
        0.2, 0.1
    )  # Green's function filter for surface vertices
    model_filter_air = Elastic(
        0.2, 0.1
    )  # Green's function filter for exterior CFD points
    filter_theta = 0.6  # Barnes-Hut opening angle for tree-accelerated N-body filter
    filter_is_acc = True  # If True, use tree-accelerated filter; otherwise brute force
    loss_weight_laplacereg = 0.01  # Weight for Laplacian smoothness regularization
    loss_weight_normalreg = 0.001  # Weight for normal consistency regularization
    #
    tri2vtx = tri2vtx.to(device)
    vtx2xyz = vtx2xyz.to(device)
    wtx2xyz = wtx2xyz.to(device)
    pix2depth_trg = pix2depth_trg.to(device)
    pix2occ_trg = pix2occ_trg.to(device)
    #
    import del_msh_dlpack.NBody.torch as NBody
    import del_msh_dlpack.IoWavefrontObj.torch as IoWavefrontObj
    import del_msh_dlpack.Vtx2Vtx.torch as Vtx2Vtx
    #
    path_dir_trg = pathlib.Path(__file__).parent.parent.parent / "target" / "out_dlpack"
    path_dir_trg.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    # Pre-compute reference Laplacian coordinates for regularization
    if loss_weight_laplacereg != 0.0:
        vtx2vtx = Vtx2Vtx.from_uniform_mesh(tri2vtx, vtx2xyz.shape[0], False)
        laplace_ini = Vtx2Vtx.GraphLaplacian.apply(*vtx2vtx, vtx2xyz).detach()

    # Pre-compute reference unit normals for normal consistency regularization
    if loss_weight_normalreg != 0.0:
        tri2unrm_ini = TriMesh3.Tri2Normal.apply(tri2vtx, vtx2xyz)
        tri2unrm_ini = (
            (tri2unrm_ini / tri2unrm_ini.norm(dim=1, keepdim=True))
            .detach()
            .requires_grad_(False)
        )
    vtx2xyz = vtx2xyz.detach().requires_grad_(True)

    from del_msh_dlpack.util_adam_uniform import UniformAdam

    optimizer = UniformAdam(params=[vtx2xyz, wtx2xyz], lr=lr)
    conv_history = []  # Track loss values for convergence monitoring

    for itr in range(0, num_itr):
        optimizer.zero_grad()
        pix2depth_src, pix2occ_src = render(
            (pix2depth_trg.shape[1], pix2depth_trg.shape[0]), views, tri2vtx, vtx2xyz
        )

        loss_depth = (pix2depth_src - pix2depth_trg).abs().mean()
        loss_occ = (pix2occ_src - pix2occ_trg).abs().mean()
        loss = loss_depth + loss_occ

        # --- Laplacian regularization: penalize deviation from initial Laplacian coords ---
        if loss_weight_laplacereg != 0.0:
            laplace_def = Vtx2Vtx.GraphLaplacian.apply(*vtx2vtx, vtx2xyz)
            loss0 = ((laplace_ini - laplace_def) ** 2).mean()
            loss += loss_weight_laplacereg * loss0

        # --- Normal regularization: penalize deviation from initial face normals ---
        if loss_weight_normalreg != 0.0:
            tri2unrm_def = TriMesh3.Tri2Normal.apply(tri2vtx, vtx2xyz)
            tri2unrm_def = tri2unrm_def / tri2unrm_def.norm(dim=1, keepdim=True)
            loss += loss_weight_normalreg * ((tri2unrm_ini - tri2unrm_def) ** 2).mean()

        print(f"itr:{itr} ### loss:{loss.cpu().item()}")
        conv_history.append(loss.cpu().item())

        #
        if itr % num_interval_save_file == 0:
            img = (pix2depth_src.detach().cpu().numpy() * 255).clip(0, 255).astype("uint8")
            path0 = (
                    path_dir_trg
                    / f"test_edgegrad_depth_src_{itr // num_interval_save_file}.png"
            )
            Image.fromarray(img).save(path0)
            #
            path0 = (
                    path_dir_trg
                    / f"test_edgegrad_depth_pnt_{itr // num_interval_save_file}.obj"
            )
            IoWavefrontObj.save_points(wtx2xyz.cpu(), path0)
            #
            path0 = (
                    path_dir_trg
                    / f"test_edgegrad_depth_bdy_{itr // num_interval_save_file}.obj"
            )
            IoWavefrontObj.save_trimesh3(tri2vtx.cpu(), vtx2xyz.cpu(), str(path0))

        # --- Backpropagate to compute gradient w.r.t. vertex positions ---
        loss.backward()
        vtx2rhs = vtx2xyz.grad.clone() / float(num_substep)
        # optimizer.step()

        # --- Apply Green's function filter (N-body smoothing) to the gradient ---
        # The filter propagates deformation from surface vertices to exterior CFD points,
        # ensuring the volume mesh deforms consistently with the surface (as in free-form
        # deformation or elasticity-based mesh morphing).
        for i_substep in range(0, num_substep):
            if filter_is_acc == False:
                # Brute-force O(N^2) filter
                vtx2lhs = NBody.filter_brute_force(
                    vtx2xyz, vtx2rhs, model_filter_body, vtx2xyz
                )
                """
                wtx2lhs = NBody.filter_brute_force(
                    vtx2xyz, vtx2rhs, model_filter_air, wtx2xyz
                )
                """
            else:
                # Tree-accelerated (Barnes-Hut) O(N log N) filter
                acc = NBody.TreeAccelerator()
                acc.initialize(vtx2xyz)
                vtx2lhs = NBody.filter_with_acceleration(
                    vtx2rhs, model_filter_body, vtx2xyz, acc, filter_theta
                )
                wtx2lhs = NBody.filter_with_acceleration(
                    vtx2rhs, model_filter_air, wtx2xyz, acc, filter_theta
                )

            # --- Update vertex positions (gradient descent step) ---
            with torch.no_grad():
                vtx2xyz.grad = vtx2lhs
                wtx2xyz.grad = wtx2lhs
            optimizer.step()

    TriMesh3.save_wavefront_obj(
        tri2vtx.cpu(), vtx2xyz.cpu(), str(path_dir_trg / "test_edgegrad_fin.obj")
    )

    end = time.perf_counter()
    print("elapsed time:", end - start)
    with open(path_dir_trg / "test_edgegrad_conv_hist.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(conv_history)



def test_match_shape_multiview(transform_world2ndc=None):
    import del_msh_dlpack.IoOff.torch as IoOff
    path_dir_asset = pathlib.Path(__file__).parent.parent.parent / "asset"
    path_dir_trg = pathlib.Path(__file__).parent.parent.parent / "target" / "out_dlpack"
    path_dir_trg.mkdir(parents=True, exist_ok=True)
    #
    pix2depth_trg, pix2occ_trg, views = make_problem()
    Image.fromarray((pix2depth_trg.numpy() * 255).clip(0, 255).astype("uint8")).save(
        path_dir_trg / f"test_edgegrad_depth_trg.png"
    )
    #
    tri2vtx, vtx2xyz = IoOff.load_tri_mesh(str(path_dir_asset / "propeller0.off"))
    TriMesh3.save_wavefront_obj(
        tri2vtx, vtx2xyz, str(path_dir_trg / "test_edgegrad_ini.obj")
    )
    wtx2xyz = sample_aabb(vtx2xyz.min(dim=0)[0], vtx2xyz.max(dim=0)[0], 1000)

    fit(torch.device("cpu"), tri2vtx, vtx2xyz.clone(), wtx2xyz.clone(), pix2depth_trg, pix2occ_trg, views)
    if torch.cuda.is_available():
        fit(torch.device("cuda"), tri2vtx, vtx2xyz.clone(), wtx2xyz.clone(), pix2depth_trg, pix2occ_trg, views)