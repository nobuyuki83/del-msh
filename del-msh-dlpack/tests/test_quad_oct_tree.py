import torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch
import del_msh_dlpack.QuadOctTree.torch

def test_01():
    # test 3D
    vtx2co = torch.rand((1000, 3))
    transform_co2unit = torch.eye(4)
    vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
    #
    (bnodes, bnode2depth, bnode2onode) = del_msh_dlpack.QuadOctTree.torch.bnodes_and_bnode2depth_and_bnode2onode(idx2morton, 3)
    num_onode = bnode2onode[-1].item() + 1
    print(num_onode)
    #
    if torch.cuda.is_available():
        (d_bnodes, d_bnode2depth, d_bnode2onode) = del_msh_dlpack.QuadOctTree.torch.bnodes_and_bnode2depth_and_bnode2onode(idx2morton.cuda(),3,10)
        assert torch.equal(bnodes, d_bnodes.cpu())
        assert torch.equal(bnode2depth, d_bnode2depth.cpu())
        assert torch.equal(bnode2onode, d_bnode2onode.cpu())

