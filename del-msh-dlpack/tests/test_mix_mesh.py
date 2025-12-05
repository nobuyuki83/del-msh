import pathlib
import del_msh_dlpack.MixMesh3.torch

def test_01():
    path = pathlib.Path(__file__).parent.parent.parent / "asset" / "cfd_mesh.txt"
    vtx2xyz, tet2vtx, pyrmd2vtx, prism2vtx = del_msh_dlpack.MixMesh3.torch.load_cfd_mesh(str(path))
    print(vtx2xyz.shape)
    print(tet2vtx.shape)
    print(pyrmd2vtx.shape)
    print(prism2vtx.shape)



