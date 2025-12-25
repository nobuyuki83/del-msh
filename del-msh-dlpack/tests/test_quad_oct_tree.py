import torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch
import del_msh_dlpack.QuadOctTree.torch
import del_msh_dlpack.OffsetArray.torch
import del_msh_dlpack.NBody.torch



def test_tree_construction():
    for num_dim in range(2,4):
        print(num_dim)
        torch.manual_seed(0)
        vtx2co = torch.rand((1000_000, num_dim))
        transform_co2unit = torch.eye(num_dim+1)
        vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
            vtx2co, transform_co2unit
        )
        (idx2vtx, idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
        idx2jdx, jdx2morton, jdx2idx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(idx2morton)
        assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(jdx2morton)
        assert torch.equal(jdx2morton, torch.unique(idx2morton.to(torch.int64)).to(torch.uint32))
        #
        tree2jdx = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
        tree2jdx.construct_from_idx2morton(jdx2morton, num_dim, True)

        if torch.cuda.is_available():
            d_vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
                vtx2co.cuda(), transform_co2unit.cuda()
            )
            assert torch.equal(vtx2morton, d_vtx2morton.cpu())
            (d_idx2vtx, d_idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton.cuda())
            d_idx2jdx, d_jdx2morton, d_jdx2idx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(d_idx2morton)
            assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(d_jdx2morton)
            d_tree2jdx = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
            d_tree2jdx.construct_from_idx2morton(d_jdx2morton, num_dim, True)
            #            
            assert torch.equal(d_jdx2morton, torch.unique(d_idx2morton.to(torch.int64)).to(torch.uint32))
            assert torch.equal(tree2jdx.bnodes, d_tree2jdx.bnodes.cpu())
            assert torch.equal(tree2jdx.bnode2depth, d_tree2jdx.bnode2depth.cpu())
            assert torch.equal(tree2jdx.bnode2onode, d_tree2jdx.bnode2onode.cpu())
            assert torch.equal(tree2jdx.idx2bnode, d_tree2jdx.idx2bnode.cpu())
            #
            assert torch.equal(tree2jdx.onodes, d_tree2jdx.onodes.cpu())
            assert torch.equal(tree2jdx.bnode2onode, d_tree2jdx.bnode2onode.cpu())
            assert torch.equal(tree2jdx.idx2bnode, d_tree2jdx.idx2bnode.cpu())
            assert torch.equal(tree2jdx.onode2depth, d_tree2jdx.onode2depth.cpu())


def test_tree_aggregation():
    '''
    TODO: this test fails when num_vtx > 17_000_000
    :return:
    '''
    num_dim = 3
    num_vdim = 3
    torch.manual_seed(0)
    num_vtx = 1_000
    vtx2co = torch.rand((num_vtx, num_dim))
    transform_co2unit = torch.eye(num_dim+1)
    vtx2val = torch.ones(size=(vtx2co.shape[0], num_vdim), dtype=torch.float32)
    #
    vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    (jdx2vtx, jdx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
    jdx2idx, idx2morton, idx2jdx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(jdx2morton)
    assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(idx2morton)
    assert torch.equal(idx2morton, torch.unique(jdx2morton.to(torch.int32)).to(torch.uint32))
    #
    tree2idx = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
    tree2idx.construct_from_idx2morton(idx2morton, num_dim, False)
    #
    idx2aggval = del_msh_dlpack.OffsetArray.torch.aggregate(idx2jdx_offset, jdx2vtx, vtx2val)
    assert torch.equal(idx2aggval.sum(dim=0), torch.full((num_vdim,), num_vtx))
    onode2aggval = tree2idx.aggregate(idx2aggval)
    assert torch.equal(onode2aggval.cpu()[0,:], torch.full((num_vdim, ), num_vtx))
    #
    if torch.cuda.is_available():
        d_vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
            vtx2co.cuda(), transform_co2unit.cuda()
        )
        assert torch.equal(vtx2morton, d_vtx2morton.cpu())
        (d_jdx2vtx, d_jdx2morton) = del_msh_dlpack.Array1D.torch.argsort(d_vtx2morton)
        assert torch.equal(jdx2vtx, d_jdx2vtx.cpu())
        assert torch.equal(jdx2morton, d_jdx2morton.cpu())
        d_jdx2idx, d_idx2morton, d_idx2jdx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(d_jdx2morton)
        assert torch.equal(jdx2idx, d_jdx2idx.cpu())
        assert torch.equal(idx2morton, d_idx2morton.cpu())
        assert torch.equal(idx2jdx_offset, d_idx2jdx_offset.cpu())
        assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(d_idx2morton)
        assert torch.equal(d_idx2morton, torch.unique(d_jdx2morton.to(torch.int32)).to(torch.uint32))
        d_tree2idx = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
        d_tree2idx.construct_from_idx2morton(d_idx2morton, num_dim, False)
        assert torch.equal(tree2idx.onodes, d_tree2idx.onodes.cpu())
        assert torch.equal(tree2idx.onode2depth, d_tree2idx.onode2depth.cpu())
        assert torch.equal(tree2idx.onode2center, d_tree2idx.onode2center.cpu())
        assert torch.equal(tree2idx.idx2onode, d_tree2idx.idx2onode.cpu())
        assert torch.equal(tree2idx.idx2center, d_tree2idx.idx2center.cpu())
        d_idx2aggval = del_msh_dlpack.OffsetArray.torch.aggregate(d_idx2jdx_offset, d_jdx2vtx, vtx2val.cuda())
        assert torch.equal(idx2aggval, d_idx2aggval.cpu())
        d_onode2aggval = d_tree2idx.aggregate(d_idx2aggval)
        assert torch.equal(onode2aggval, d_onode2aggval.cpu())














