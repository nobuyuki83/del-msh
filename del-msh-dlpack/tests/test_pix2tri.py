import pathlib

from PIL import Image
import numpy as np
import torch


import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.TriMesh3.torch as TriMesh3
import render_util


def test_lambertian_shading_phong():
    path_dir = pathlib.Path(__file__).parent.parent.parent / "target" / "out_dlpack"
    path_dir.mkdir(parents=True, exist_ok=True)
    #
    from test_pix2depth import example1

    tri2vtx, vtx2xyz, transform_world2ndc, img_shape = example1()
    vtx2xyz.requires_grad_(True)
    vtx2xyz.grad = None
    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = Pix2Tri.by_raycasting(
        tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
    )
    vtx2nrm = TriMesh3.make_vtx2normal(tri2vtx.int(), vtx2xyz)
    pix2rgb = render_util.render_lambertian_shading_phong(
        tri2vtx,
        vtx2xyz,
        vtx2nrm,
        transform_ndc2world,
        [0.0, 0.0, 1.0],
        [0.8, 1.0, 0.9],
        pix2tri,
    )
    torch.random.manual_seed(0)
    pix2trg = torch.rand_like(pix2rgb)
    loss = torch.nn.functional.mse_loss(pix2rgb, pix2trg)
    loss.backward()
    dldw_vtx2xyz = vtx2xyz.grad.clone()
    #
    img = (pix2rgb.detach().numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(img).save(path_dir / "pix2tri2.png")
    #
    if torch.cuda.is_available():
        d_tri2vtx = tri2vtx.cuda()
        d_vtx2xyz = vtx2xyz.detach().cuda()
        d_vtx2xyz.grad = None
        d_vtx2xyz.requires_grad_(True)
        d_transform_ndc2world = transform_ndc2world.cuda()
        d_pix2trg = pix2trg.cuda()
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
        d_pix2rgb = render_util.render_lambertian_shading_phong(
            d_tri2vtx,
            d_vtx2xyz,
            d_vtx2nrm,
            d_transform_ndc2world,
            [0.0, 0.0, 1.0],
            [0.8, 1.0, 0.9],
            d_pix2tri,
        )
        assert (d_pix2rgb.cpu() - pix2rgb).abs().max() < 5.0e-6
        d_loss = torch.nn.functional.mse_loss(d_pix2rgb, d_pix2trg)
        d_loss.backward()
        d_dldw_vtx2xyz = d_vtx2xyz.grad.clone()

        print((d_dldw_vtx2xyz.cpu() - dldw_vtx2xyz).abs().max())


def _colormap_jet_np(t: np.ndarray) -> np.ndarray:
    """Vectorized jet colormap for t in [0, 1], returns (N, 3) uint8."""
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def test_pix2tri():
    import del_msh_dlpack.IoWavefrontObj.torch as IoWavefrontObj
    import del_msh_dlpack.Vtx2Xyz.torch as Vtx2Xyz

    path_dir_asset = pathlib.Path(__file__).parent.parent.parent / "asset"
    path_dir_out = pathlib.Path(__file__).parent.parent.parent / "target" / "out_dlpack"
    path_dir_out.mkdir(parents=True, exist_ok=True)

    IMG_RES = 256
    img_shape = (IMG_RES, IMG_RES)

    tri2vtx, vtx2xyz = IoWavefrontObj.load_tri_mesh3(path_dir_asset / "bunny_50k.obj")
    vtx2xyz = Vtx2Xyz.normalize(vtx2xyz, 1.0)
    num_tri = tri2vtx.shape[0]

    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)

    import del_msh_dlpack.Mat44.torch as Mat44

    NUM_ITR = 4
    for i_itr in range(NUM_ITR):
        z_trans = -2.0 * float(NUM_ITR - i_itr - 1) / float(NUM_ITR)
        transform0 = Mat44.camera_perspective_blender(1.0, 35.0, 0.1, 3.0, True)
        transform1 = Mat44.from_translation(0.0, 0.0, z_trans)
        transform_world2ndc = transform0 @ transform1
        transform_ndc2world = transform_world2ndc.inverse().contiguous()

        pix2tri_raycast = Pix2Tri.by_raycasting(
            tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape
        )

        pix2tri_rasterization = Pix2Tri.by_rasterization(
            tri2vtx, vtx2xyz, transform_world2ndc, img_shape
        )

        uint32_max = torch.iinfo(torch.uint32).max
        ray_bg = pix2tri_raycast == uint32_max
        rst_bg = pix2tri_rasterization == uint32_max

        assert torch.equal(ray_bg, rst_bg), (
            f"iter {i_itr}: foreground/background mismatch, "
            f"diff pixels: {(ray_bg != rst_bg).sum().item()}"
        )

        num_mismatch = int((pix2tri_raycast != pix2tri_rasterization).sum())
        assert num_mismatch <= 10, (
            f"iter {i_itr}: too many triangle-identity mismatches: {num_mismatch}"
        )

        # Save colorized PNG
        pix2tri_flat = pix2tri_raycast.reshape(-1).numpy()  # uint32 array
        pix2rgb = np.zeros((IMG_RES * IMG_RES, 3), dtype=np.uint8)
        fg_mask = pix2tri_flat != np.uint32(uint32_max)
        if fg_mask.any():
            t = pix2tri_flat[fg_mask].astype(np.float32) / float(num_tri)
            pix2rgb[fg_mask] = _colormap_jet_np(t)
        img = pix2rgb.reshape(IMG_RES, IMG_RES, 3)
        Image.fromarray(img).save(path_dir_out / f"pix2tri_{i_itr}.png")

        if torch.cuda.is_available():
            d_tri2vtx = tri2vtx.cuda()
            d_vtx2xyz = vtx2xyz.cuda()
            d_bvhnodes, d_bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(d_tri2vtx, d_vtx2xyz)
            d_transform_world2ndc = transform_world2ndc.cuda()
            d_transform_ndc2world = transform_ndc2world.cuda()
            d_pix2tri_raycast = Pix2Tri.by_raycasting(
                d_tri2vtx, d_vtx2xyz, d_bvhnodes, d_bvhnode2aabb, d_transform_ndc2world, img_shape
            )
            d0 = (d_pix2tri_raycast.cpu() != pix2tri_raycast).sum().item()
            #print("raycast cpu-gpu mismatch:", d0)
            assert d0 == 0
            #
            d_pix2tri_rasterization = Pix2Tri.by_rasterization(
                d_tri2vtx, d_vtx2xyz, d_transform_world2ndc, img_shape
            )
            d1 = (d_pix2tri_rasterization.cpu() != pix2tri_rasterization).sum().item()
            #print("rasterization cpu-gpu mismatch:", d1)
            assert d1 == 0
