import torch
from .. import util_torch

def from_edge2vtx_of_tri2vtx_with_vtx2vtx(
        edge2vtx: torch.Tensor,
        tri2vtx: torch.Tensor,
        vtx2idx_offset: torch.Tensor,
        idx2vtx: torch.Tensor,
        edge2tri: torch.Tensor):
    """Compute the two triangles adjacent to each edge (edge-to-triangle adjacency).

    For each edge, finds the (at most two) triangles that share it, using the
    vertex-to-vertex adjacency as an acceleration structure.

    Args:
        edge2vtx: (num_edge, 2) uint32 - edge connectivity
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2idx_offset: (num_vtx+1,) uint32 - offset array of vertex-to-vertex adjacency
        idx2vtx: (num_edge,) uint32 - adjacent vertex indices (from vtx2vtx)
        edge2tri: (num_edge, 2) uint32 - output: triangle indices per edge (modified in-place)
    """
    num_edge = edge2vtx.shape[0]
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    device = edge2vtx.device
    #
    util_torch.assert_shape_dtype_device(edge2vtx, (num_edge, 2), torch.uint32, device)
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2idx_offset, (num_vtx + 1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(edge2tri, (num_edge, 2), torch.uint32, device)
    #
    from .. import Edge2Elem

    Edge2Elem.from_edge2vtx_of_tri2vtx_with_vtx2vtx(
        edge2vtx.__dlpack__(),
        tri2vtx.__dlpack__(),
        vtx2idx_offset.__dlpack__(),
        idx2vtx.__dlpack__(),
        edge2tri.__dlpack__()
    )
