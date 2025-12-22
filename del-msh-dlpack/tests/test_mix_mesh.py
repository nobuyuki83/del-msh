import torch
import pathlib
import del_msh_dlpack.MixMesh3.torch

def test_01():
    path = pathlib.Path(__file__).parent.parent.parent / "asset" / "cfd_mesh.txt"
    vtx2xyz, tet2vtx, pyrmd2vtx, prism2vtx = del_msh_dlpack.MixMesh3.torch.load_cfd_mesh(str(path))
    print(vtx2xyz.shape)
    print(tet2vtx.shape)
    print(pyrmd2vtx.shape)
    print(prism2vtx.shape)
    print(tet2vtx.dtype)
    path_vtk = pathlib.Path(__file__).parent.parent.parent / "target" / "mix_mesh.vtk"
    del_msh_dlpack.MixMesh3.torch.save_vtk(
        str(path_vtk),
        vtx2xyz,
        tet2vtx.to(torch.uint32),
        pyrmd2vtx.to(torch.uint32),
        prism2vtx.to(torch.uint32))





