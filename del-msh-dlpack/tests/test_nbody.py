import torch
import del_msh_dlpack.NBody.torch

def test_spoisson_3d():
    num_dim = 3
    torch.manual_seed(0)
    num_vtx = 10_000
    vtx2xyz = torch.rand((num_vtx, num_dim)) * 3.0 - 1.0
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3), dtype=torch.float32) * 2.0 - 1.0
    num_wtx = 100_00
    wtx2xyz = torch.rand((num_wtx, num_dim)) * 10.0 - 5.0
    wtx2lhs0 = del_msh_dlpack.NBody.torch.screened_poisson(vtx2xyz, vtx2rhs, 0.1, 0.0001, wtx2xyz)
    #
    acc = del_msh_dlpack.NBody.torch.TreeAccelerator()
    acc.initialize(vtx2xyz)
    wtx2lhs1 = del_msh_dlpack.NBody.torch.screened_poisson_with_acceleration(vtx2rhs, 0.1, 0.0001, wtx2xyz, acc, 0.2)
    diff = (wtx2lhs0-wtx2lhs1).abs().max().item()
    scale = wtx2lhs0.abs().max().item()
    assert diff/scale < 0.006

    if not torch.cuda.is_available():
        return



def test_elastic_3d():
    num_dim = 3
    torch.manual_seed(0)
    num_vtx = 10_000
    vtx2xyz = torch.rand((num_vtx, num_dim)) * 3.0 - 1.0
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3), dtype=torch.float32) * 2.0 - 1.0
    num_wtx = 100_000
    wtx2xyz = torch.rand((num_wtx, num_dim)) * 10.0 - 5.0
    wtx2lhs0 = del_msh_dlpack.NBody.torch.elastic(vtx2xyz, vtx2rhs, 0.1, 0.0001, wtx2xyz)
    #
    acc = del_msh_dlpack.NBody.torch.TreeAccelerator()
    acc.initialize(vtx2xyz)
    #
    print(wtx2lhs0.abs().max())