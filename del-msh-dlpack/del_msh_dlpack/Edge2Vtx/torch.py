import torch
#
from .. import util_torch

def from_vtx2vtx(vtx2idx_offset: torch.Tensor, idx2vtx: torch.Tensor, edge2vtx: torch.Tensor):
    num_vtx = vtx2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_edge = edge2vtx.shape[0]
    device = vtx2idx_offset.device
    #
    assert num_edge == num_idx
    util_torch.assert_shape_dtype_device(vtx2idx_offset, (num_vtx + 1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(edge2vtx, (num_edge, 2), torch.uint32, device)
    #
    from .. import Edge2Vtx

    Edge2Vtx.from_vtx2vtx(
        vtx2idx_offset.__dlpack__(),
        idx2vtx.__dlpack__(),
        edge2vtx.__dlpack__()
    )
