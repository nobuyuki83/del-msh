import pathlib


from PIL import Image
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
