import torch
from .. import _CapsuleAsDLPack


def load_tri_mesh(path: str):
    """Load a triangle mesh from an OFF file.

    Args:
        path: path to the .off file
    Returns:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    """
    from .. import IoOff

    cap_tri2vtx, cap_vtx2xyz = IoOff.load_tri_mesh(path)
    tri2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tri2vtx))
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz))
    return tri2vtx, vtx2xyz
