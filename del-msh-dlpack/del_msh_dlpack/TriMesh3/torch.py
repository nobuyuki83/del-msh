import torch
#
from .. import util_torch
from .. import _CapsuleAsDLPack


def make_tri2centroid(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor) -> torch.Tensor:
    """compute centroids of triangles

    Args:b
        tri2vtx: (num_tri, 3) uint32
        vtx2xyz: (num_vtx, 3) float32
    Returns:
        tri2centroid: (num_tri, 3) float32
    """
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    idx = tri2vtx.long()
    return (vtx2xyz[idx[:, 0]] + vtx2xyz[idx[:, 1]] + vtx2xyz[idx[:, 2]]) / 3.0


def make_tri2normal(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor):
    """Compute the unit normal vector of each triangle.

    Args:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    Returns:
        tri2nrm: (num_tri, 3) float32 - unit normal per triangle
    """
    num_tri = tri2vtx.shape[0]
    num_vtx = vtx2xyz.shape[0]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
        # print(device, stream_ptr)
    #
    tri2nrm = torch.empty(size=(num_tri, 3), dtype=torch.float32, device=device)
    from .. import TriMesh3

    TriMesh3.tri2normal(
        util_torch.to_dlpack_safe(tri2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(tri2nrm, stream_ptr),
        stream_ptr=stream_ptr,
    )
    return tri2nrm


def bwd_tri2normal(
    tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor, dw_tri2nrm: torch.Tensor
):
    """Backward pass of make_tri2normal: propagate loss gradient to vertex positions.

    Args:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
        dw_tri2nrm: (num_tri, 3) float32 - loss gradient w.r.t. triangle normals
    Returns:
        dw_vtx2xyz: (num_vtx, 3) float32 - loss gradient w.r.t. vertex positions
    """
    num_vtx = vtx2xyz.shape[0]
    num_tri = tri2vtx.shape[0]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri, 3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx, 3), torch.float32, device)
    util_torch.assert_shape_dtype_device(dw_tri2nrm, (num_tri, 3), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    dw_vtx2xyz = torch.empty(size=(num_vtx, 3), dtype=torch.float32, device=device)
    from .. import TriMesh3

    TriMesh3.bwd_tri2normal(
        util_torch.to_dlpack_safe(tri2vtx, stream_ptr),
        util_torch.to_dlpack_safe(vtx2xyz, stream_ptr),
        util_torch.to_dlpack_safe(dw_tri2nrm, stream_ptr),
        util_torch.to_dlpack_safe(dw_vtx2xyz, stream_ptr),
        stream_ptr=stream_ptr,
    )
    return dw_vtx2xyz


class Tri2Normal(torch.autograd.Function):
    """Differentiable triangle normal computation as a torch.autograd.Function."""

    @staticmethod
    def forward(ctx, tri2vtx, vtx2xyz):
        ctx.save_for_backward(tri2vtx, vtx2xyz)
        return make_tri2normal(tri2vtx.detach(), vtx2xyz.detach())

    @staticmethod
    def backward(ctx, dw_tri2nrm):
        tri2vtx, vtx2xyz = ctx.saved_tensors
        dw_vtx2xyz = bwd_tri2normal(tri2vtx.detach(), vtx2xyz.detach(), dw_tri2nrm)
        return None, dw_vtx2xyz


def load_nastran(
        path_file: str):
    """Load a triangle mesh from a Nastran file.

    Args:
        path_file: path to the Nastran file
    Returns:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    """
    from .. import TriMesh3

    cap_tri2vtx, cap_vtx2xyz = TriMesh3.load_nastran(path_file)
    tri2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tri2vtx))
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz))
    return tri2vtx, vtx2xyz


def save_wavefront_obj(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor, path_file: str):
    """Save a triangle mesh to a Wavefront OBJ file.

    Args:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity (CPU only)
        vtx2xyz: (num_vtx, 3) float32 - vertex positions (CPU only)
        path_file: output file path
    """
    assert tri2vtx.device.type == "cpu"
    assert vtx2xyz.device.type == "cpu"

    from .. import TriMesh3

    TriMesh3.save_wavefront_obj(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        path_file
    )


def torus(major_raidus: float, minor_radius: float, ndiv_major: int, ndiv_minor: int):
    """Generate a torus triangle mesh.

    Args:
        major_raidus: radius from the center of the tube to the center of the torus
        minor_radius: radius of the tube
        ndiv_major: number of divisions around the major axis
        ndiv_minor: number of divisions around the tube
    Returns:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    """
    from .. import TriMesh3

    cap_tri2vtx, cap_vtx2xyz = TriMesh3.torus(major_raidus, minor_radius, ndiv_major, ndiv_minor)
    tri2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tri2vtx)).clone()
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz)).clone()
    return tri2vtx, vtx2xyz


def sphere(raidus: float, ndiv_longtitude: int, ndiv_latitude: int):
    """Generate a sphere triangle mesh.

    Args:
        raidus: radius of the sphere
        ndiv_longtitude: number of divisions along longitude
        ndiv_latitude: number of divisions along latitude
    Returns:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    """
    from .. import TriMesh3

    cap_tri2vtx, cap_vtx2xyz = TriMesh3.sphere(raidus, ndiv_longtitude, ndiv_latitude)
    tri2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tri2vtx)).clone()
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz)).clone()
    return tri2vtx, vtx2xyz


def make_bvhnodes_bvhnode2aabb(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor):
    """Build a BVH (Bounding Volume Hierarchy) for a triangle mesh.

    Constructs BVH nodes and their AABBs using Morton-code sorting of triangle centroids.

    Args:
        tri2vtx: (num_tri, 3) uint32 - triangle connectivity
        vtx2xyz: (num_vtx, 3) float32 - vertex positions
    Returns:
        bvhnodes: (2*num_tri-1, 3) uint32 - BVH node data (left, right, parent)
        bvhnode2aabb: (2*num_tri-1, 6) float32 - axis-aligned bounding box per node
    """
    num_vtx = vtx2xyz.shape[0]
    num_tri = tri2vtx.shape[0]
    device = tri2vtx.device
    #
    util_torch.assert_shape_dtype_device(tri2vtx, (num_tri,3), torch.uint32, device)
    util_torch.assert_shape_dtype_device(vtx2xyz, (num_vtx,3), torch.float32, device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    tri2centroid = make_tri2centroid(tri2vtx, vtx2xyz)
    from ..Mat44.torch import from_fit_vtx2xyz_into_unit_cube
    transform_co2unit = from_fit_vtx2xyz_into_unit_cube(tri2centroid)
    #
    from ..Mortons.torch import make_vtx2morton_from_vtx2co
    tri2morton = make_vtx2morton_from_vtx2co(tri2centroid, transform_co2unit)
    tri2tri = torch.arange(num_tri, dtype=torch.int32, device=device).to(dtype=torch.uint32)
    from ..Array1D.torch import argsort
    idx2tri, idx2morton = argsort(tri2morton)
    from ..Mortons.torch import make_bvhnodes_from_sorted_mortons
    bvhnodes = make_bvhnodes_from_sorted_mortons(idx2tri, idx2morton)
    num_bvhnodes = bvhnodes.shape[0]
    #
    bvhnode2aabb = torch.empty((num_bvhnodes, 6), dtype=torch.float32, device=device)
    vtx2xyz1 = torch.zeros((0,3), dtype=torch.float32, device=device)
    from .. import TriMesh3
    TriMesh3.make_bvhnode2aabb_from_bvhnodes(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        vtx2xyz1.__dlpack__(),
        bvhnodes.__dlpack__(),
        bvhnode2aabb.__dlpack__(),
        stream_ptr=stream_ptr,
    )
    return bvhnodes, bvhnode2aabb


def make_edge2vtx(tri2vtx: torch.Tensor, num_vtx: int):
    assert tri2vtx.ndim == 2 and tri2vtx.shape[1] == 3
    assert tri2vtx.is_contiguous()
    assert tri2vtx.dtype == torch.uint32
    #
    from del_msh_dlpack.Vtx2Vtx.torch import from_uniform_mesh
    vtx2idx_offset, idx2vtx = from_uniform_mesh(tri2vtx, num_vtx, False)
    edge2vtx = torch.empty((idx2vtx.shape[0], 2), dtype=torch.uint32)
    from del_msh_dlpack.Edge2Vtx.torch import from_vtx2vtx
    from_vtx2vtx(vtx2idx_offset, idx2vtx, edge2vtx)
    return edge2vtx


def make_edge2tri(tri2vtx: torch.Tensor, num_vtx: int, edge2vtx: torch.Tensor):
    assert tri2vtx.ndim == 2 and tri2vtx.shape[1] == 3
    assert tri2vtx.is_contiguous()
    assert tri2vtx.dtype == torch.uint32
    assert edge2vtx.ndim == 2 and edge2vtx.shape[1] == 2
    assert edge2vtx.is_contiguous()
    assert edge2vtx.dtype == torch.uint32
    assert tri2vtx.device == edge2vtx.device
    #
    from del_msh_dlpack.Vtx2Elem.torch import from_uniform_mesh
    vtx2jdx_offset, jdx2tri = from_uniform_mesh(tri2vtx, num_vtx)
    edge2tri = torch.empty((edge2vtx.shape[0], 2), dtype=torch.uint32)
    from del_msh_dlpack.Edge2Elem.torch import from_edge2vtx_of_tri2vtx_with_vtx2vtx
    from_edge2vtx_of_tri2vtx_with_vtx2vtx(edge2vtx, tri2vtx, vtx2jdx_offset, jdx2tri, edge2tri)
    return edge2tri
