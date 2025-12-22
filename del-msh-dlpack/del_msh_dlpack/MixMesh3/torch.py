import torch
from .. import _CapsuleAsDLPack

def load_cfd_mesh(path: str):
    from .. import MixMesh3
    cap_vtx2xyz, cap_tet2vtx, cap_pyrmd2vtx, cap_prism2vtx = MixMesh3.load_cfd_mesh(path)
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz))
    tet2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tet2vtx))
    pyrmd2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_pyrmd2vtx))
    prism2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_prism2vtx))
    return vtx2xyz, tet2vtx, pyrmd2vtx, prism2vtx


def save_vtk(
    path_file: str,
    vtx2xyz: torch.Tensor,
    tet2vtx: torch.Tensor,
    pyrmd2vtx: torch.Tensor,
    prism2vtx: torch.Tensor):
    num_vtx = vtx2xyz.shape[0]
    num_tet = tet2vtx.shape[0]
    num_pyrmd = pyrmd2vtx.shape[0]
    num_prism = prism2vtx.shape[0]
    #
    assert vtx2xyz.shape == (num_vtx,3)
    assert vtx2xyz.dtype == torch.float32
    assert vtx2xyz.device.type == "cpu"
    #
    from .. import MixMesh3
    MixMesh3.save_vtk(
        path_file,
        vtx2xyz.__dlpack__(),
        tet2vtx.__dlpack__(),
        pyrmd2vtx.__dlpack__(),
        prism2vtx.__dlpack__())
