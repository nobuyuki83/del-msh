import torch
from .. import util_torch
from .. import _CapsuleAsDLPack

def make_elem2volume(
    elem2idx_offset: torch.Tensor,
    idx2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor):
    """Compute the volume of each polyhedral element.

    Args:
        elem2idx_offset: (num_elem+1,) uint32 - offset array into idx2vtx per element
        idx2vtx: (num_idx,) uint32 - concatenated vertex indices for all elements
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    Returns:
        elem2volume: (num_elem,) float32 - volume of each element
    """
    #
    num_elem = elem2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    device = elem2idx_offset.device
    #
    util_torch.assert_shape_dtype_device(elem2idx_offset, (num_elem+1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx,3), torch.float32, device)
    #
    elem2volume = torch.empty(num_elem, dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import PolyhedronMesh

    PolyhedronMesh.elem2volume(
        util_torch.to_dlpack_safe(elem2idx_offset, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(elem2volume, stream_ptr),
        stream_ptr
    )
    return elem2volume


def make_elem2centroid(
    elem2idx_offset: torch.Tensor,
    idx2vtx: torch.Tensor,
    vtx2xyz: torch.Tensor) -> torch.Tensor:
    """Compute the center of each polyhedral element.

    Args:
        elem2idx_offset: (num_elem+1,) uint32 - offset array into idx2vtx per element
        idx2vtx: (num_idx,) uint32 - concatenated vertex indices for all elements
        vtx2xyz: (num_vtx, num_dim) float32 - vertex positions
    Returns:
        elem2center: (num_elem, num_dim) float32 - center of each element
    """
    #
    num_elem = elem2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_vtx, num_dim = vtx2xyz.shape
    device = elem2idx_offset.device
    #
    util_torch.assert_shape_dtype_device(elem2idx_offset, (num_elem+1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, num_dim), torch.float32, device)
    #
    elem2center = torch.empty((num_elem, num_dim), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import PolyhedronMesh

    PolyhedronMesh.elem2center(
        util_torch.to_dlpack_safe(elem2idx_offset, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(elem2center, stream_ptr),
        stream_ptr
    )
    return elem2center


def make_bvhnodes_bvhnode2aabb(
        elem2idx_offset: torch.Tensor, 
        idx2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor):
    num_vtx = vtx2xyz.shape[0]
    num_elem = elem2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    device = elem2idx_offset.device
    vtx2xyz = vtx2xyz.detach()
    #
    util_torch.assert_shape_dtype_device(elem2idx_offset, (num_elem+1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device) 
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx,3), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    elem2centroid = make_elem2centroid(elem2idx_offset, idx2vtx, vtx2xyz)
    from ..Mat44.torch import from_fit_vtx2xyz_into_unit_cube
    transform_co2unit = from_fit_vtx2xyz_into_unit_cube(elem2centroid)
    #
    from ..Mortons.torch import make_vtx2morton_from_vtx2co
    elem2morton = make_vtx2morton_from_vtx2co(elem2centroid, transform_co2unit)
    from ..Array1D.torch import argsort
    idx2elem, idx2morton = argsort(elem2morton)
    from ..Mortons.torch import make_bvhnodes_from_sorted_mortons
    bvhnodes = make_bvhnodes_from_sorted_mortons(idx2elem, idx2morton)
    num_bvhnodes = bvhnodes.shape[0]
    #
    bvhnode2aabb = torch.empty((num_bvhnodes, 6), dtype=torch.float32, device=device)
    from .. import PolyhedronMesh
    PolyhedronMesh.bvhnode2aabb_from_bvhnodes(
        util_torch.to_dlpack_safe(elem2idx_offset, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(bvhnodes, stream_ptr),
        util_torch.to_dlpack_safe(bvhnode2aabb, stream_ptr),
        stream_ptr
    )
    return bvhnodes, bvhnode2aabb


def search_elem_contain_points(
        elem2idx_offset: torch.Tensor,
        idx2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor,
        bvhnodes: torch.Tensor,
        bvhnode2aabb: torch.Tensor,
        wtx2xyz: torch.Tensor):
    """For each query point find the nearest polyhedron element and its interpolation weights.

    Args:
        elem2idx_offset: (num_elem+1,) uint32 - CSR offset array
        idx2vtx: (num_idx,) uint32 - concatenated vertex indices for all elements
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
        bvhnodes: (num_bvhnode, 3) uint32 - BVH node data
        bvhnode2aabb: (num_bvhnode, 6) float32 - AABB per BVH node
        wtx2xyz: (num_wtx, 3) float32 - query points
    Returns:
        wtx2elem: (num_wtx,) uint32 - nearest element index per query point
        wtx2param: (num_wtx, 3) float32 - parametric coordinates within the nearest element:
                   tet: (r0,r1,r2) with r3=1-r0-r1-r2 implicit;
                   pyramid: (r,s,t) with r,s in [0,1], t in [0,1];
                   prism: (r,s,t) with r,s>=0, r+s<=1, t in [0,1]
    """
    num_wtx = wtx2xyz.shape[0]
    device = wtx2xyz.device
    #
    util_torch.assert_shape_dtype_device(wtx2xyz, (num_wtx, 3), torch.float32, device)
    #
    wtx2elem = torch.empty(num_wtx, dtype=torch.uint32, device=device)
    wtx2param = torch.empty((num_wtx, 3), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import PolyhedronMesh
    PolyhedronMesh.search_elem_contain_points(
        util_torch.to_dlpack_safe(bvhnodes, stream_ptr),
        util_torch.to_dlpack_safe(bvhnode2aabb, stream_ptr),
        util_torch.to_dlpack_safe(elem2idx_offset, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(wtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(wtx2elem, stream_ptr),
        util_torch.to_dlpack_safe(wtx2param, stream_ptr),
        stream_ptr
    )
    return wtx2elem, wtx2param

def interpolate_values_at_points(
        elem2idx_offset: torch.Tensor,
        idx2vtx: torch.Tensor,
        vtx2value: torch.Tensor,
        wtx2elem: torch.Tensor,
        wtx2param: torch.Tensor) -> torch.Tensor:
    """Interpolate per-vertex values at query points using pre-computed element membership.

    Args:
        elem2idx_offset: (num_elem+1,) uint32 - CSR offset array
        idx2vtx: (num_idx,) uint32 - concatenated vertex indices
        vtx2value: (num_vtx, num_value_dim) float32 - per-vertex values to interpolate
        wtx2elem: (num_wtx,) uint32 - element index per query (u32::MAX if outside mesh)
        wtx2param: (num_wtx, 3) float32 - parametric coords from search_elem_contain_points
    Returns:
        wtx2value: (num_wtx, num_value_dim) float32 - interpolated values (0 for outside points)
    """
    num_elem = elem2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_vtx, num_value_dim = vtx2value.shape
    num_wtx = wtx2elem.shape[0]
    device = elem2idx_offset.device
    #
    util_torch.assert_shape_dtype_device(elem2idx_offset, (num_elem + 1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2value, (num_vtx, num_value_dim), torch.float32, device)
    util_torch.assert_shape_dtype_device(wtx2elem, (num_wtx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(wtx2param, (num_wtx, 3), torch.float32, device)
    #
    wtx2value_out = torch.zeros((num_wtx, num_value_dim), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import PolyhedronMesh
    PolyhedronMesh.interpolate_values_at_points(
        util_torch.to_dlpack_safe(elem2idx_offset, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2value, stream_ptr),
        util_torch.to_dlpack_safe(wtx2elem, stream_ptr),
        util_torch.to_dlpack_safe(wtx2param, stream_ptr),
        util_torch.to_dlpack_safe(wtx2value_out, stream_ptr),
        stream_ptr
    )
    return wtx2value_out


def subdivide(
        elem2idx_offset: torch.Tensor,
        idx2vtx: torch.Tensor,
        vtx2xyz: torch.Tensor):
    """Subdivide a mixed polyhedron mesh once.

    Args:
        elem2idx_offset: (num_elem+1,) uint32 - CSR offset array
        idx2vtx: (num_idx,) uint32 - concatenated vertex indices for all elements
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    Returns:
        elem2idx_offset: (num_new_elem+1,) uint32
        idx2vtx: (num_new_idx,) uint32
        vtx2xyz: (num_new_vtx, 3) float32
    """
    num_elem = elem2idx_offset.shape[0] - 1
    num_idx = idx2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    device = elem2idx_offset.device
    #
    util_torch.assert_shape_dtype_device(elem2idx_offset, (num_elem + 1,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(idx2vtx, (num_idx,), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import PolyhedronMesh
    cap0, cap1, cap2 = PolyhedronMesh.subdivide(
        util_torch.to_dlpack_safe(elem2idx_offset, stream_ptr),
        util_torch.to_dlpack_safe(idx2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        stream_ptr
    )
    return (
        torch.from_dlpack(_CapsuleAsDLPack(cap0)),
        torch.from_dlpack(_CapsuleAsDLPack(cap1)),
        torch.from_dlpack(_CapsuleAsDLPack(cap2)),
    )

