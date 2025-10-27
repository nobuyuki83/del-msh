import torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch
import del_msh_dlpack.QuadOctTree.torch


def test_02():
    for num_dim in range(2,3):
        print(num_dim)
        torch.manual_seed(0)
        vtx2co = torch.rand((10_000_000, num_dim))
        transform_co2unit = torch.eye(num_dim+1)
        vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
            vtx2co, transform_co2unit
        )
        (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
        idx2jdx, jdx2morton, jdx2idx_offdset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(idx2morton)
        assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(jdx2morton)
        assert torch.equal(jdx2morton, torch.unique(idx2morton.to(torch.int32)).to(torch.uint32))
        #
        tree = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
        tree.construct_from_idx2morton(jdx2morton, num_dim, True)
        if torch.cuda.is_available():
            print("gpu")
            d_vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
                vtx2co.cuda(), transform_co2unit.cuda()
            )
            (d_idx2vtx, d_idx2morton) = del_msh_dlpack.Array1D.torch.argsort(d_vtx2morton)
            d_idx2jdx, d_jdx2morton, d_jdx2idx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(d_idx2morton)
            assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(d_jdx2morton)
            assert torch.equal(d_jdx2morton, torch.unique(d_idx2morton.to(torch.int32)).to(torch.uint32))
            d_tree = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
            d_tree.construct_from_idx2morton(d_jdx2morton, num_dim, True)
            #
            assert torch.equal(tree.bnodes, d_tree.bnodes.cpu())
            assert torch.equal(tree.bnode2depth, d_tree.bnode2depth.cpu())
            assert torch.equal(tree.bnode2onode, d_tree.bnode2onode.cpu())
            assert torch.equal(tree.idx2bnode, d_tree.idx2bnode.cpu())
            #
            assert torch.equal(tree.onodes, d_tree.onodes.cpu())
            assert torch.equal(tree.bnode2onode, d_tree.bnode2onode.cpu())
            assert torch.equal(tree.idx2bnode, d_tree.idx2bnode.cpu())
            assert torch.equal(tree.onode2depth, d_tree.onode2depth.cpu())
            print("gpu end")










