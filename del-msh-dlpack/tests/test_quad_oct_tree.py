import torch
import del_msh_dlpack.Mortons.torch
import del_msh_dlpack.Array1D.torch
import del_msh_dlpack.QuadOctTree.torch
import del_msh_dlpack.OffsetArray.torch


def test_01():
    for num_dim in range(2,4):
        print(num_dim)
        torch.manual_seed(0)
        vtx2co = torch.rand((1_000_000, num_dim))
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
            d_vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
                vtx2co.cuda(), transform_co2unit.cuda()
            )
            assert torch.equal(vtx2morton, d_vtx2morton.cpu())
            (d_idx2vtx, d_idx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton.cuda())
            d_idx2jdx, d_jdx2morton, d_jdx2idx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(d_idx2morton)
            assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(d_jdx2morton)
            d_tree = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
            d_tree.construct_from_idx2morton(d_jdx2morton, num_dim, True)
            #            
            assert torch.equal(d_jdx2morton, torch.unique(d_idx2morton.to(torch.int32)).to(torch.uint32))            
            assert torch.equal(tree.bnodes, d_tree.bnodes.cpu())
            assert torch.equal(tree.bnode2depth, d_tree.bnode2depth.cpu())
            assert torch.equal(tree.bnode2onode, d_tree.bnode2onode.cpu())
            assert torch.equal(tree.idx2bnode, d_tree.idx2bnode.cpu())
            #
            assert torch.equal(tree.onodes, d_tree.onodes.cpu())
            assert torch.equal(tree.bnode2onode, d_tree.bnode2onode.cpu())
            assert torch.equal(tree.idx2bnode, d_tree.idx2bnode.cpu())
            assert torch.equal(tree.onode2depth, d_tree.onode2depth.cpu())


def test_02():
    if not torch.cuda.is_available():
        return
    '''
    TODO: this test fails when num_vtx > 17_000_000
    :return:
    '''
    num_dim = 3
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_vtx = 16_000_000
    vtx2co = torch.rand((num_vtx, num_dim), device=device)
    transform_co2unit = torch.eye(num_dim+1, device=device)
    vtx2morton = del_msh_dlpack.Mortons.torch.vtx2morton_from_vtx2co(
        vtx2co, transform_co2unit
    )
    (jdx2vtx, jdx2morton) = del_msh_dlpack.Array1D.torch.argsort(vtx2morton)
    jdx2idx, idx2morton, idx2jdx_offdset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(jdx2morton)
    assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(idx2morton)
    assert torch.equal(idx2morton, torch.unique(jdx2morton.to(torch.int32)).to(torch.uint32))
    #
    tree = del_msh_dlpack.QuadOctTree.torch.QuadOctTree()
    tree.construct_from_idx2morton(idx2morton, num_dim, False)
    #
    num_vdim = 8
    vtx2val = torch.ones(size=(vtx2co.shape[0], num_vdim), dtype=torch.float32, device=device)
    idx2aggval = del_msh_dlpack.OffsetArray.torch.aggregate(idx2jdx_offdset, jdx2vtx, vtx2val)
    assert torch.equal(idx2aggval.sum(dim=0).cpu(), torch.full((num_vdim,), num_vtx))
    #
    onode2aggval = tree.aggregate(idx2aggval)
    assert torch.equal(onode2aggval.cpu()[0,:], torch.full((num_vdim, ), num_vtx))








