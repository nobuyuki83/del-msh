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