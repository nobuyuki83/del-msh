import torch
from .. import util_torch

def bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode(
    idx2morton: torch.Tensor,
    num_dim,
    max_depth: int | None = None):
    #
    if max_depth is None:
        max_depth = 32 // num_dim
    num_idx = idx2morton.shape[0]
    num_bnode = num_idx - 1
    device = idx2morton.device
    #
    assert idx2morton.dtype == torch.uint32
    #
    bnodes = torch.empty((num_bnode,3), device=device, dtype=torch.uint32)
    bnode2depth = torch.empty((num_bnode,), device=device, dtype=torch.uint32)
    bnode2onode = torch.empty((num_bnode,), device=device, dtype=torch.uint32)
    idx2bnode = torch.empty((num_idx,), device=device, dtype=torch.uint32)
    #
    idx2bnode.fill_(2**32 - 1) # the maximum of uint32
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import QuadOctTree

    QuadOctTree.bnodes_and_bnode2depth_and_bnode2onode(
                    util_torch.to_dlpack_safe(idx2morton, stream_ptr),
                    num_dim,
                    max_depth,
                    util_torch.to_dlpack_safe(bnodes, stream_ptr),
                    util_torch.to_dlpack_safe(bnode2depth, stream_ptr),
                    util_torch.to_dlpack_safe(bnode2onode, stream_ptr),
                    util_torch.to_dlpack_safe(idx2bnode, stream_ptr),
                    stream_ptr)

    return bnodes, bnode2depth, bnode2onode, idx2bnode


def make_tree_from_binary_radix_tree(
    bnodes: torch.Tensor,
    bnode2onode: torch.Tensor,
    bnode2depth: torch.Tensor,
    idx2bnode: torch.Tensor,
    idx2morton: torch.Tensor,
    num_dim: int,
    max_depth: int | None = None,
):
    num_bnode = bnodes.shape[0]
    num_idx = num_bnode + 1
    if max_depth is None:
        max_depth = 32 // num_dim
    #
    nlink = 1 + 2**num_dim
    num_onode = bnode2onode[-1].item() + 1
    device = bnodes.device
    #
    onodes = torch.empty((num_onode, nlink), device=device, dtype=torch.uint32)
    onode2depth = torch.empty((num_onode, ), device=device, dtype=torch.uint32)
    onode2center = torch.empty((num_onode, num_dim), device=device, dtype=torch.float32)
    idx2onode = torch.empty((num_idx, ), device=device, dtype=torch.uint32)
    idx2center= torch.empty((num_idx, num_dim), device=device, dtype=torch.float32)
    #
    onode2depth.fill_(0)
    onodes.fill_(2**32 - 1) # the maximum of uint32
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import QuadOctTree
    QuadOctTree.make_tree_from_binary_radix_tree(
        util_torch.to_dlpack_safe(bnodes, stream_ptr),
        util_torch.to_dlpack_safe(bnode2onode, stream_ptr),
        util_torch.to_dlpack_safe(bnode2depth, stream_ptr),
        util_torch.to_dlpack_safe(idx2bnode, stream_ptr),
        util_torch.to_dlpack_safe(idx2morton, stream_ptr),
        num_dim,
        max_depth,
        util_torch.to_dlpack_safe(onodes, stream_ptr),
        util_torch.to_dlpack_safe(onode2depth, stream_ptr),
        util_torch.to_dlpack_safe(onode2center, stream_ptr),
        util_torch.to_dlpack_safe(idx2onode, stream_ptr),
        util_torch.to_dlpack_safe(idx2center, stream_ptr),
        stream_ptr
    )
    return onodes, onode2depth, onode2center, idx2onode, idx2center


class QuadOctTree:
    def __init__(self):
        pass

    def construct_from_idx2morton(self, idx2morton: torch.Tensor, num_dim: int, is_save_intermediate = False):
        self.num_dim = num_dim
        (bnodes, bnode2depth, bnode2onode, idx2bnode) \
            = bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode(
            idx2morton, num_dim)
        (self.onodes, self.onode2depth, self.onode2center, self.idx2onode, self.idx2center) \
            = make_tree_from_binary_radix_tree(
            bnodes, bnode2onode, bnode2depth,
            idx2bnode, idx2morton, num_dim)
        if is_save_intermediate:
            self.bnodes = bnodes
            self.bnode2depth = bnode2depth
            self.bnode2onode = bnode2onode
            self.idx2bnode = idx2bnode

    def aggregate(self, idx2val) -> torch.Tensor:
        num_vdim = idx2val.shape[1]
        num_idx = self.idx2onode.shape[0]
        num_onode = self.onodes.shape[0]
        device = self.onodes.device
        #
        assert idx2val.shape == (num_idx, num_vdim) and idx2val.device == device and idx2val.dtype == torch.float32
        #
        onode2aggval = torch.zeros(size=(num_onode, num_vdim), dtype=torch.float32, device=device)
        #
        stream_ptr = 0
        if device.type == "cuda":
            torch.cuda.set_device(device)
            stream_ptr = torch.cuda.current_stream(device).cuda_stream
        #
        from .. import QuadOctTree
        QuadOctTree.aggregate(
            util_torch.to_dlpack_safe(idx2val,stream_ptr),
            util_torch.to_dlpack_safe(self.idx2onode, stream_ptr),
            util_torch.to_dlpack_safe(self.onodes, stream_ptr),
            util_torch.to_dlpack_safe(onode2aggval, stream_ptr),
            stream_ptr
        )
        return onode2aggval








