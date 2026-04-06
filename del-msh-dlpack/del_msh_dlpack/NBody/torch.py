import torch
from .. import util_torch
import del_msh_dlpack.QuadOctTree.torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch
import del_msh_dlpack.Mat44.torch as Mat44
import del_msh_dlpack.OffsetArray.torch
from .. import NBody


def filter_brute_force(
        vtx2co: torch.Tensor,
        vtx2rhs: torch.Tensor,
        model: NBody.Model,
        wtx2co: torch.Tensor):
    """Evaluate an N-body filter using brute-force O(N*M) summation.

    For each query point in wtx2co, sums contributions from all source points
    in vtx2co weighted by vtx2rhs using the given kernel model.

    Args:
        vtx2co: (num_vtx, 3) float32 - source point positions
        vtx2rhs: (num_vtx, 3) float32 - source point values (right-hand side)
        model: NBody.Model - kernel model defining the interaction
        wtx2co: (num_wtx, 3) float32 - query point positions
    Returns:
        wtx2lhs: (num_wtx, 3) float32 - accumulated filter output per query point
    """
    num_vtx = vtx2co.shape[0]
    num_wtx = wtx2co.shape[0]
    device = vtx2co.device
    #
    util_torch.assert_shape_dtype_device(vtx2co, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(vtx2rhs, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(wtx2co, (num_wtx, 3), torch.float32, device)
    #
    wtx2lhs = torch.zeros(size=(num_wtx, 3), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import NBody

    NBody.filter_brute_force(
        util_torch.to_dlpack_safe(vtx2co, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        util_torch.to_dlpack_safe(wtx2co, stream_ptr),
        util_torch.to_dlpack_safe(wtx2lhs, stream_ptr),
        model,
        stream_ptr
    )

    return wtx2lhs



class TreeAccelerator:
    """Acceleration structure for N-body filtering using a quad/oct-tree with Morton codes.

    Organizes source points into a hierarchical tree to enable fast approximate
    N-body summation via the `filter_with_acceleration` function.
    """

    def __init__(self):
        self.tree2idx = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()

    def initialize(self, vtx2xyz: torch.Tensor):
        """Build the tree from a set of source points.

        Args:
            vtx2xyz: (num_vtx, 3) float32 - source point positions
        """
        device = vtx2xyz.device
        num_dim = vtx2xyz.shape[1]
        assert vtx2xyz.dtype == torch.float
        self.transform_world2unit = Mat44.from_transfrom_world2unit(vtx2xyz, device)
        vtx2morton = del_msh_dlpack.Mortons.torch.make_vtx2morton_from_vtx2co(
            vtx2xyz, self.transform_world2unit)
        (self.jdx2vtx, jdx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
        jdx2idx, idx2morton, self.idx2jdx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(jdx2morton)
        self.tree2idx.construct_from_idx2morton(idx2morton, num_dim, False)
        self.onode2cgunit = self.construct_center_of_gravity(self.idx2jdx_offset, self.jdx2vtx, vtx2xyz, self.transform_world2unit)
        self.vtx2xyz = vtx2xyz

    def construct_center_of_gravity(self, idx2jdx_offset, jdx2vtx, vtx2xyz, transform_world2unit):
        """Compute the center of gravity of each tree node in unit-cube coordinates."""
        num_vtx = jdx2vtx.shape[0]
        device = idx2jdx_offset.device
        assert vtx2xyz.shape[0] == num_vtx
        idx2aggxyz = del_msh_dlpack.OffsetArray.torch.aggregate(idx2jdx_offset, jdx2vtx, vtx2xyz)
        idx2nvtx = del_msh_dlpack.OffsetArray.torch.aggregate(
            idx2jdx_offset, jdx2vtx,
            torch.ones(size=(num_vtx, 1), dtype=torch.float, device=device))
        onode2aggxyz = self.tree2idx.aggregate(idx2aggxyz)
        onode2nvtx = self.tree2idx.aggregate(idx2nvtx)
        assert onode2nvtx[0] == float(num_vtx)
        onode2cgxyz = onode2aggxyz / onode2nvtx
        ones = torch.ones((onode2cgxyz.shape[0], 1), dtype=torch.float, device=onode2cgxyz.device)
        onode2cgxyzw = torch.cat([onode2cgxyz, ones], dim=1)  # (N,4)
        return (onode2cgxyzw @ transform_world2unit.T)[:, 0:3].clone()


def filter_with_acceleration(
        vtx2rhs: torch.Tensor,
        model: NBody.Model,
        wtx2xyz: torch.Tensor,
        acc: TreeAccelerator,
        theta: float):
    """Evaluate an N-body filter using tree-based acceleration (Barnes-Hut style).

    Approximates the full O(N*M) summation by grouping distant source points into
    tree nodes. The theta parameter controls accuracy vs. speed trade-off.

    Args:
        vtx2rhs: (num_vtx, 3) float32 - source point values (right-hand side)
        model: NBody.Model - kernel model defining the interaction
        wtx2xyz: (num_wtx, 3) float32 - query point positions
        acc: TreeAccelerator - pre-built acceleration structure over source points
        theta: float - opening angle threshold; smaller = more accurate but slower
    Returns:
        wtx2lhs: (num_wtx, 3) float32 - accumulated filter output per query point
    """
    num_vtx = vtx2rhs.shape[0]
    num_wtx = wtx2xyz.shape[0]
    num_idx = acc.idx2jdx_offset.shape[0] - 1
    num_onode = acc.tree2idx.onodes.shape[0]
    device = vtx2rhs.device
    #
    util_torch.assert_shape_dtype_device(wtx2xyz, (num_wtx,3), torch.float32, device)
    util_torch.assert_shape_dtype_device(vtx2rhs, (num_vtx,3), torch.float32, device)
    util_torch.assert_shape_dtype_device(acc.vtx2xyz, (num_vtx,3), torch.float32, device)
    util_torch.assert_shape_dtype_device(acc.jdx2vtx, (num_vtx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(acc.tree2idx.idx2onode, (num_idx, ), torch.uint32, device)
    util_torch.assert_shape_dtype_device(acc.tree2idx.onodes, (num_onode, 9), torch.uint32, device)
    util_torch.assert_shape_dtype_device(acc.tree2idx.onode2depth, (num_onode, ), torch.uint32, device)
    util_torch.assert_shape_dtype_device(acc.tree2idx.onode2center, (num_onode, 3), torch.float32, device)
    #
    idx2aggrhs = del_msh_dlpack.OffsetArray.torch.aggregate(acc.idx2jdx_offset, acc.jdx2vtx, vtx2rhs)
    onode2aggrhs = acc.tree2idx.aggregate(idx2aggrhs)
    wtx2lhs = torch.zeros((num_wtx, 3), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import NBody

    NBody.filter_with_acceleration(
        util_torch.to_dlpack_safe(acc.vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        util_torch.to_dlpack_safe(wtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(wtx2lhs, stream_ptr),
        model,
        util_torch.to_dlpack_safe(acc.transform_world2unit.permute(1, 0).contiguous().view(-1), stream_ptr),
        util_torch.to_dlpack_safe(acc.idx2jdx_offset, stream_ptr),
        util_torch.to_dlpack_safe(acc.jdx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(acc.tree2idx.onodes, stream_ptr),
        util_torch.to_dlpack_safe(acc.tree2idx.onode2center, stream_ptr),
        util_torch.to_dlpack_safe(acc.tree2idx.onode2depth, stream_ptr),
        util_torch.to_dlpack_safe(acc.onode2cgunit, stream_ptr),
        util_torch.to_dlpack_safe(onode2aggrhs, stream_ptr),
        theta,
        stream_ptr)
    return wtx2lhs


