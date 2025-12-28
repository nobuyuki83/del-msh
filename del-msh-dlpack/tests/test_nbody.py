import torch
import del_msh_dlpack.NBody.torch

def test_spoisson_3d():
    num_dim = 3
    torch.manual_seed(0)
    num_vtx = 1_000
    vtx2xyz = torch.rand((num_vtx, num_dim)) * 3.0 - 1.0
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3), dtype=torch.float32) * 2.0 - 1.0
    num_wtx = 1_000
    wtx2xyz = torch.rand((num_wtx, num_dim)) * 10.0 - 5.0
    model = del_msh_dlpack.NBody.ScreenedPoisson(0.1, 0.0001)
    theta = 0.2
    wtx2lhs0 = del_msh_dlpack.NBody.torch.filter_brute_force(vtx2xyz, vtx2rhs, model, wtx2xyz)
    #
    acc = del_msh_dlpack.NBody.torch.TreeAccelerator()
    acc.initialize(vtx2xyz)
    wtx2lhs1 = del_msh_dlpack.NBody.torch.filter_with_acceleration(vtx2rhs, model, wtx2xyz, acc, theta)
    diff = (wtx2lhs0-wtx2lhs1).abs().max().item()
    scale = wtx2lhs0.abs().max().item()
    assert diff/scale < 0.006
    #
    if torch.cuda.is_available():
        d_acc = del_msh_dlpack.NBody.torch.TreeAccelerator()
        d_acc.initialize(vtx2xyz.cuda())
        d_wtx2lhs1 = del_msh_dlpack.NBody.torch.filter_with_acceleration(
            vtx2rhs.cuda(), model, wtx2xyz.cuda(), d_acc,
            theta)
        diff = (d_wtx2lhs1.cpu()-wtx2lhs1).abs().max().item()
        print(diff)
        assert diff/scale < 1.0e-6




def test_elastic_3d():
    num_dim = 3
    torch.manual_seed(0)
    num_vtx = 1_000
    vtx2xyz = torch.rand((num_vtx, num_dim)) * 3.0 - 1.0
    vtx2rhs = torch.rand(size=(vtx2xyz.shape[0], 3), dtype=torch.float32) * 2.0 - 1.0
    num_wtx = 1_000
    wtx2xyz = torch.rand((num_wtx, num_dim)) * 10.0 - 5.0
    model = del_msh_dlpack.NBody.Elastic(0.1, 0.0001)
    theta = 0.2
    wtx2lhs0 = del_msh_dlpack.NBody.torch.filter_brute_force(vtx2xyz, vtx2rhs, model, wtx2xyz)
    #
    acc = del_msh_dlpack.NBody.torch.TreeAccelerator()
    acc.initialize(vtx2xyz)
    wtx2lhs1 = del_msh_dlpack.NBody.torch.filter_with_acceleration(vtx2rhs, model, wtx2xyz, acc, theta)
    diff = (wtx2lhs0-wtx2lhs1).abs().max().item()
    scale = wtx2lhs0.abs().max().item()
    print("elastic", diff, scale, diff/scale)
    assert diff/scale < 0.03
    if torch.cuda.is_available():
        d_acc = del_msh_dlpack.NBody.torch.TreeAccelerator()
        d_acc.initialize(vtx2xyz.cuda())
        d_wtx2lhs1 = del_msh_dlpack.NBody.torch.filter_with_acceleration(
            vtx2rhs.cuda(), model, wtx2xyz.cuda(), d_acc,
            theta)
        diff = (d_wtx2lhs1.cpu()-wtx2lhs1).abs().max().item()
        print(diff/scale)
        assert diff/scale < 1.4e-6

