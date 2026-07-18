import pathlib
#
from PIL import Image
import torch
#
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.Pix2Tri.torch as Pix2Tri
import del_msh_dlpack.TriMesh3.torch as TriMesh3


def test_lambertian():
    from test_pix2depth import example1
    tri2vtx, vtx2xyz, transform_world2ndc, img_shape = example1()
    vtx2nrm = TriMesh3.make_vtx2normal(tri2vtx.int(), vtx2xyz)
    transform_ndc2world = transform_world2ndc.inverse().contiguous()
    #
    bvhnodes, bvhnode2aabb = TriMesh3.make_bvhnodes_bvhnode2aabb(tri2vtx, vtx2xyz)
    pix2tri = Pix2Tri.by_raycasting(tri2vtx, vtx2xyz, bvhnodes, bvhnode2aabb, transform_ndc2world, img_shape)
    background = pix2tri == torch.iinfo(torch.uint32).max
    #
    pix2nrm = Pix2Tri.interpolate_fwd(pix2tri, tri2vtx, vtx2xyz, vtx2nrm, transform_ndc2world)
    background_nrm = pix2nrm.new_tensor([0.0, 0.0, 1.0])
    pix2nrm = torch.where(
        background.unsqueeze(-1),
        background_nrm,
        pix2nrm,
    )
    pix2nrm = torch.nn.functional.normalize(
        pix2nrm,
        dim=-1,
    )
    #
    img = (((pix2nrm + 1.) * 0.5).detach().numpy() * 255).clip(0, 255).astype('uint8')
    path0 = pathlib.Path(__file__).parent.parent.parent / "target" / "del_msh_dlpack__pix2tri1.png"
    path0.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path0)
    #
    light_dir = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3)
    base_color = torch.tensor([0.8, 1.0, 0.9]).view(1, 1, 3)
    pix2ndotl = torch.clamp(
        torch.sum(pix2nrm * light_dir, dim=-1, keepdim=True),
        min=0.0,
    )
    pix2diffuse = pix2ndotl * base_color
    pix2rgb = pix2diffuse
    background_color = torch.ones_like(pix2rgb)
    pix2rgb = torch.where(background.unsqueeze(-1), background_color, pix2rgb)
    #
    img = (pix2rgb.detach().numpy() * 255).clip(0, 255).astype('uint8')
    path0 = pathlib.Path(__file__).parent.parent.parent / "target" / "del_msh_dlpack__pix2tri2.png"
    path0.parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path0)
