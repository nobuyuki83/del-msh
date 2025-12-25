import torch
from .. import util_torch
import del_msh_dlpack.QuadOctTree.torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch


def screened_poisson(
        vtx2co: torch.Tensor,
        vtx2rhs: torch.Tensor,
        lambda_param: float,
        epsilon: float,
        wtx2co: torch.Tensor):
    num_vtx = vtx2co.shape[0]
    num_wtx = wtx2co.shape[0]
    device = vtx2co.device
    #
    assert vtx2co.shape == (num_vtx, 3) and vtx2rhs.dtype == torch.float32
    assert vtx2rhs.shape == (num_vtx, 3) and vtx2rhs.device == device and vtx2rhs.dtype == torch.float32
    assert wtx2co.shape == (num_wtx, 3) and wtx2co.device == device and wtx2co.dtype == torch.float32
    #
    wtx2lhs = torch.zeros(size=(num_wtx, 3), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import NBody

    NBody.screened_poisson(
        util_torch.to_dlpack_safe(vtx2co, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        util_torch.to_dlpack_safe(wtx2co, stream_ptr),
        util_torch.to_dlpack_safe(wtx2lhs, stream_ptr),
        lambda_param,
        epsilon,
        stream_ptr
    )

    return wtx2lhs


def elastic(
        vtx2co: torch.Tensor,
        vtx2rhs: torch.Tensor,
        nu: float,
        epsilon: float,
        wtx2co: torch.Tensor):
    num_vtx = vtx2co.shape[0]
    num_wtx = wtx2co.shape[0]
    device = vtx2co.device
    #
    assert vtx2co.shape == (num_vtx, 3) and vtx2rhs.dtype == torch.float32
    assert vtx2rhs.shape == (num_vtx, 3) and vtx2rhs.device == device and vtx2rhs.dtype == torch.float32
    assert wtx2co.shape == (num_wtx, 3) and wtx2co.device == device and wtx2co.dtype == torch.float32
    #
    wtx2lhs = torch.zeros(size=(num_wtx, 3), dtype=torch.float32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import NBody

    NBody.elastic(
        util_torch.to_dlpack_safe(vtx2co, stream_ptr),
        util_torch.to_dlpack_safe(vtx2rhs, stream_ptr),
        util_torch.to_dlpack_safe(wtx2co, stream_ptr),
        util_torch.to_dlpack_safe(wtx2lhs, stream_ptr),
        nu,
        epsilon,
        stream_ptr
    )

    return wtx2lhs


def mat4_from_translate(t):
    return torch.Tensor([
        [1., 0., 0., t[0]],
        [0., 1., 0., t[1]],
        [0., 0., 1., t[2]],
        [0., 0., 0., 1.]
    ])

def mat4_from_uniform_scale(s):
    return torch.Tensor([
        [s, 0., 0., 0.],
        [0., s, 0., 0.],
        [0., 0., s, 0.],
        [0., 0., 0., 1.]
    ])

def mat4_from_transfrom_world2unit(vtx2xyz):
    xyz_min = vtx2xyz.min(dim=0).values
    xyz_max = vtx2xyz.max(dim=0).values
    xyz_len = xyz_max - xyz_min
    scale = 1.0/xyz_len.max().item()
    xyz_center = (xyz_min + xyz_max)*0.5
    print(xyz_min, xyz_max, xyz_len, xyz_center, scale)
    m1 = mat4_from_translate(-xyz_center)
    m2 = mat4_from_uniform_scale(scale)
    m3 = mat4_from_translate([0.5, 0.5, 0.5])
    return m3 @ m2 @ m1

def vtx2xyz_transform_affine(vtx2xyz, transform):
    ones = torch.ones((vtx2xyz.shape[0], 1), dtype=torch.float, device=vtx2xyz.device)
    vtx2xyzw = torch.cat([vtx2xyz, ones], dim=1)  # (N,4)
    return (vtx2xyzw @ transform.T)[:, 0:3].clone()


class TreeAccelerator:

    def __init__(self):
        self.tree2idx = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()

    def initialize(self, vtx2xyz):
        self.transform_world2unit = mat4_from_transfrom_world2unit(vtx2xyz)
        num_dim = vtx2xyz.shape[1]
        self.vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
            vtx2xyz, self.transform_world2unit)
        (self.jdx2vtx, self.jdx2morton) = del_msh_dlpack.Array1D.torch.argsort(self.vtx2morton)
        self.jdx2idx, idx2morton, self.idx2jdx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(self.jdx2morton)
        assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(idx2morton)
        assert torch.equal(idx2morton, torch.unique(self.jdx2morton.to(torch.int32)).to(torch.uint32))
        self.tree2idx.construct_from_idx2morton(idx2morton, num_dim, False)
        self.onode2cgunit = self.constract_center_of_gravity(self.idx2jdx_offset, self.jdx2vtx, vtx2xyz, self.transform_world2unit)
        self.vtx2xyz = vtx2xyz

    def constract_center_of_gravity(self, idx2jdx_offset, jdx2vtx, vtx2xyz, transform_world2unit):
        num_vtx = jdx2vtx.shape[0]
        assert vtx2xyz.shape[0] == num_vtx
        idx2aggxyz = del_msh_dlpack.OffsetArray.torch.aggregate(idx2jdx_offset, jdx2vtx, vtx2xyz)
        idx2nvtx = del_msh_dlpack.OffsetArray.torch.aggregate(
            idx2jdx_offset, jdx2vtx,
            torch.ones(size=(num_vtx, 1), dtype=torch.float))
        onode2aggxyz = self.tree2idx.aggregate(idx2aggxyz)
        onode2nvtx = self.tree2idx.aggregate(idx2nvtx)
        assert onode2nvtx[0] == float(num_vtx)
        onode2cgxyz = onode2aggxyz / onode2nvtx
        ones = torch.ones((onode2cgxyz.shape[0], 1), dtype=torch.float, device=onode2cgxyz.device)
        onode2cgxyzw = torch.cat([onode2cgxyz, ones], dim=1)  # (N,4)
        return (onode2cgxyzw @ transform_world2unit.T)[:, 0:3].clone()


def screened_poisson_with_acceleration(
        acc: TreeAccelerator,
        vtx2rhs: torch.Tensor,
        lambda_: float ,
        eps: float,
        wtx2xyz: TreeAccelerator,
        theta: float):
    num_vtx = vtx2rhs.shape[0]
    assert acc.vtx2xyz.shape == (num_vtx, 3)
    assert acc.vtx2morton.shape == (num_vtx, )
    assert acc.jdx2vtx.shape == (num_vtx, )
    assert acc.jdx2idx.shape == (num_vtx, )
    assert acc.jdx2morton.shape == (num_vtx, )
    idx2aggrhs = del_msh_dlpack.OffsetArray.torch.aggregate(acc.idx2jdx_offset, acc.jdx2vtx, vtx2rhs)
    onode2aggrhs = acc.tree2idx.aggregate(idx2aggrhs)




