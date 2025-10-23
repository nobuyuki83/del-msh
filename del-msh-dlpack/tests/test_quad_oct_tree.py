import torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch
import del_msh_dlpack.QuadOctTree.torch

def test_01():
    # test 3D
    vtx2co = torch.rand((10, 3))
    transform_co2unit = torch.eye(4)
    vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
    #
    (bnodes, bnode2depth, bnode2onode, idx2bnode) \
     = del_msh_dlpack.QuadOctTree.torch.bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode(
      idx2morton, 3)
    num_onode = bnode2onode[-1].item() + 1
    print(num_onode)
    (onodes, onode2depth, onode2center, idx2onode, idx2center) \
      = del_msh_dlpack.QuadOctTree.torch.make_tree_from_binary_radix_tree(
       bnodes,
       bnode2onode,
       bnode2depth,
       idx2bnode,
       idx2morton,
       3)
    #
    if torch.cuda.is_available():
        d_idx2morton = idx2morton.cuda()
        (d_bnodes, d_bnode2depth, d_bnode2onode, d_idx2bnode) \
          = del_msh_dlpack.QuadOctTree.torch.bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode(
          idx2morton.cuda(),3,10)
        assert torch.equal(bnodes, d_bnodes.cpu())
        assert torch.equal(bnode2depth, d_bnode2depth.cpu())
        assert torch.equal(bnode2onode, d_bnode2onode.cpu())
        assert torch.equal(idx2bnode, d_idx2bnode.cpu())
        (d_onodes, d_onode2depth, d_onode2center, d_idx2onode, d_idx2center) \
          = del_msh_dlpack.QuadOctTree.torch.make_tree_from_binary_radix_tree(
          d_bnodes,
          d_bnode2onode,
          d_bnode2depth,
          d_idx2bnode,
          d_idx2morton,
          3)
        assert torch.equal(onodes, d_onodes.cpu())
        assert torch.equal(bnode2onode, d_bnode2onode.cpu())
        assert torch.equal(idx2bnode, d_idx2bnode.cpu())
        assert torch.equal(onode2depth, d_onode2depth.cpu())







