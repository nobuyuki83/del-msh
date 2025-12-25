import torch
import del_msh_dlpack.NBody.torch


def test_barnes_hut():
    num_dim = 3
    torch.manual_seed(0)
    num_vtx = 1_000
    vtx2xyz = torch.rand((num_vtx, num_dim)) * 3.0 - 1.0
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3), dtype=torch.float32)
    vtx2lhs0 = del_msh_dlpack.NBody.torch.screened_poisson(vtx2xyz, vtx2rhs, 0.1, 0.0001, vtx2xyz)
    #
    acc = del_msh_dlpack.NBody.torch.TreeAccelerator()
    acc.initialize(vtx2xyz)
    del_msh_dlpack.NBody.torch.screened_poisson_with_acceleration(acc, vtx2rhs, 0.1, 0.0001, vtx2xyz, 0.3)




    if not torch.cuda.is_available():
        return