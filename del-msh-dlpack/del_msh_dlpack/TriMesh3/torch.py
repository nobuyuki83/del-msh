import torch
from .. import util_torch
from .. import _CapsuleAsDLPack


def tri2centroid(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor) -> torch.Tensor:
    """compute centroids of triangles

    Args:
        tri2vtx: (num_tri, 3) uint32
        vtx2xyz: (num_vtx, 3) float32
    Returns:
        tri2centroid: (num_tri, 3) float32
    """
    assert tri2vtx.shape[1] == 3 and tri2vtx.dtype == torch.uint32
    assert vtx2xyz.shape[1] == 3 and vtx2xyz.dtype == torch.float32
    idx = tri2vtx.long()
    return (vtx2xyz[idx[:, 0]] + vtx2xyz[idx[:, 1]] + vtx2xyz[idx[:, 2]]) / 3.0


def tri2normal(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor):
    num_tri = tri2vtx.shape[0]
    device = tri2vtx.device
    assert len(tri2vtx.shape) == 2
    assert tri2vtx.shape[1] == 3
    assert tri2vtx.dtype == torch.uint32
    #
    assert len(vtx2xyz.shape) == 2
    assert vtx2xyz.shape[1] == 3
    assert vtx2xyz.dtype == torch.float32
    assert vtx2xyz.device == device, "vtx2xyz should be on the same device as tri2vtx"
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
    num_vtx = vtx2xyz.shape[0]
    num_tri = tri2vtx.shape[0]
    device = tri2vtx.device
    #
    assert tri2vtx.shape == (num_tri,3) and tri2vtx.dtype == torch.uint32
    assert vtx2xyz.shape == (num_vtx,3) and vtx2xyz.dtype == torch.float32 and vtx2xyz.device == device
    assert dw_tri2nrm.shape == (num_tri,3) and dw_tri2nrm.dtype == torch.float32 and dw_tri2nrm.device == device
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
    @staticmethod
    def forward(ctx, tri2vtx, vtx2xyz):
        ctx.save_for_backward(tri2vtx, vtx2xyz)
        return tri2normal(tri2vtx.detach(), vtx2xyz.detach())

    @staticmethod
    def backward(ctx, dw_tri2nrm):
        tri2vtx, vtx2xyz = ctx.saved_tensors
        dw_vtx2xyz = bwd_tri2normal(tri2vtx.detach(), vtx2xyz.detach(), dw_tri2nrm)
        return None, dw_vtx2xyz


def load_nastran(
        path_file: str):
    from .. import TriMesh3

    cap_tri2vtx, cap_vtx2xyz = TriMesh3.load_nastran(path_file)
    tri2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tri2vtx))
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz))
    return tri2vtx, vtx2xyz


def save_wavefront_obj(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor, path_file: str):
    assert tri2vtx.device.type == "cpu"
    assert vtx2xyz.device.type == "cpu"

    from .. import TriMesh3

    TriMesh3.save_wavefront_obj(
        tri2vtx.__dlpack__(),
        vtx2xyz.__dlpack__(),
        path_file
    )


def torus(major_raidus: float, minor_radius: float, ndiv_major: int, ndiv_minor: int):
    from .. import TriMesh3

    cap_tri2vtx, cap_vtx2xyz = TriMesh3.torus(major_raidus, minor_radius, ndiv_major, ndiv_minor)
    tri2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tri2vtx)).clone()
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz)).clone()
    return tri2vtx, vtx2xyz


def sphere(raidus: float, ndiv_longtitude: int, ndiv_latitude: int):
    from .. import TriMesh3

    cap_tri2vtx, cap_vtx2xyz = TriMesh3.sphere(raidus, ndiv_longtitude, ndiv_latitude)
    tri2vtx = torch.from_dlpack(_CapsuleAsDLPack(cap_tri2vtx)).clone()
    vtx2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_vtx2xyz)).clone()
    return tri2vtx, vtx2xyz



def bvhnodes(tri2vtx: torch.Tensor, vtx2xyz: torch.Tensor):
    num_vtx = vtx2xyz.shape[0]
    num_tri = tri2vtx.shape[0]
    device = tri2vtx.device
    util_torch.check_tensor(tri2vtx, (num_tri,3), torch.uint32, device)
    util_torch.check_tensor(vtx2xyz, (num_vtx,3), torch.float32, device)
    #
    tri2centroid = tri2centroid(tri2vtx, vtx2xyz)
    from ..Vtx2Xyz.torch import normalize_to_unit_cube
    transform_co2unit = normalize_to_unit_cube(vtx2xyz)
    #
    from ..Mortons import torch
    tri2morton = torch.vtx2morton_from_vtx2co(tri2centroid, transform_co2unit)
    tri2tri = torch.arrange(num_tri, dtype=torch.uint32, device=device)
    bvhnodes = torch.make_bvh(tri2tri, tri2morton)