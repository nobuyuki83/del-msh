import torch
from .. import util_torch
from .. import _CapsuleAsDLPack

def load_cfd_mesh(path: str):
    """Load a CFD mesh from file.

    Args:
        path: path to the CFD mesh file
    Returns:
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
        tet2vtx: (num_tet, 4) uint32 - tetrahedron connectivity
        pyrmd2vtx: (num_pyrmd, 5) uint32 - pyramid connectivity
        prism2vtx: (num_prism, 6) uint32 - prism connectivity
    """
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
    """Save a mixed-element mesh to a VTK file.

    Args:
        path_file: output file path
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
        tet2vtx: (num_tet, 4) uint32 - tetrahedron connectivity
        pyrmd2vtx: (num_pyrmd, 5) uint32 - pyramid connectivity
        prism2vtx: (num_prism, 6) uint32 - prism connectivity
    """
    #
    num_vtx = vtx2xyz.shape[0]
    num_tet = tet2vtx.shape[0]
    num_pyrmd = pyrmd2vtx.shape[0]
    num_prism = prism2vtx.shape[0]
    #
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, torch.device("cpu"))
    util_torch.assert_shape_dtype_device(tet2vtx, (num_tet, 4), torch.uint32, torch.device("cpu"))
    util_torch.assert_shape_dtype_device(pyrmd2vtx, (num_pyrmd, 5), torch.uint32, torch.device("cpu"))
    util_torch.assert_shape_dtype_device(prism2vtx, (num_prism, 6), torch.uint32, torch.device("cpu"))
    #
    from .. import MixMesh3
    MixMesh3.save_vtk(
        path_file,
        vtx2xyz.__dlpack__(),
        tet2vtx.__dlpack__(),
        pyrmd2vtx.__dlpack__(),
        prism2vtx.__dlpack__())


def to_polyhedral_mesh(
    tet2vtx: torch.Tensor,
    pyrmd2vtx: torch.Tensor,
    prism2vtx: torch.Tensor):
    """Convert a mixed-element mesh to a polyhedral mesh (offset-array format).

    Args:
        tet2vtx: (num_tet, 4) uint32 - tetrahedron connectivity
        pyrmd2vtx: (num_pyrmd, 5) uint32 - pyramid connectivity
        prism2vtx: (num_prism, 6) uint32 - prism connectivity
    Returns:
        elem2idx_offset: (num_elem+1,) uint32 - offset array into idx2vtx per element
        idx2vtx: (num_idx,) uint32 - concatenated vertex indices for all elements
    """
    #
    device = tet2vtx.device
    num_tet = tet2vtx.shape[0]
    num_pyrmd = pyrmd2vtx.shape[0]
    num_prism = prism2vtx.shape[0]
    num_elem = num_tet + num_pyrmd + num_prism
    num_idx = num_tet * 4 + num_pyrmd * 5 + num_prism * 6
    #
    util_torch.assert_shape_dtype_device(tet2vtx, (num_tet, 4), torch.uint32, device)
    util_torch.assert_shape_dtype_device(pyrmd2vtx, (num_pyrmd, 5), torch.uint32, device)
    util_torch.assert_shape_dtype_device(prism2vtx, (num_prism, 6), torch.uint32, device)
    #
    elem2idx_offset = torch.empty(size=(num_elem+1,), device=device, dtype=torch.uint32)
    idx2vtx = torch.empty(size=(num_idx,), device=device, dtype=torch.uint32)
    #
    from .. import MixMesh3
    MixMesh3.to_polyhedron_mesh(
        tet2vtx.__dlpack__(),
        pyrmd2vtx.__dlpack__(),
        prism2vtx.__dlpack__(),
        elem2idx_offset.__dlpack__(),
        idx2vtx.__dlpack__())

    assert elem2idx_offset[-1] == idx2vtx.shape[0]

    return elem2idx_offset, idx2vtx