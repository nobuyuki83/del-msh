import torch
from .. import util_torch

def binary_radix_tree_and_depth(
    idx2morton: torch.Tensor,
    num_dim,
    max_depth: int | None = None):
    if max_depth is None:
        max_depth = 32 // num_dim
    print(max_depth)
    num_idx = idx2morton.shape[0]
    num_bnode = num_idx - 1
    device = idx2morton.device
    #
    bnodes = torch.empty((num_bnode,3), device=device, dtype=torch.uint32)
    bnode2depth = torch.empty((num_bnode,), device=device, dtype=torch.uint32)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import QuadOctTree

    QuadOctTree.binary_radix_tree_and_depth(
                    util_torch.to_dlpack_safe(idx2morton, stream_ptr),
                    num_dim,
                    max_depth,
                    util_torch.to_dlpack_safe(bnodes, stream_ptr),
                    util_torch.to_dlpack_safe(bnode2depth, stream_ptr),
                    stream_ptr)

    return bnodes, bnode2depth