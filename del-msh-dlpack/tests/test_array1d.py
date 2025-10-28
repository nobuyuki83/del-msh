import torch
import del_msh_dlpack.Array1D.torch

def test_has_duplicate_in_sorted_array():
    vals = torch.arange(100).to(torch.uint32)
    assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(vals)
    vals = torch.cat((vals, torch.tensor([30], dtype=torch.uint32)))
    vals, _ = torch.sort(vals)
    assert del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(vals)
    if torch.cuda.is_available():
        vals = torch.arange(100).to(torch.uint32)
        assert not del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(vals.cuda())
        vals = torch.cat((vals, torch.tensor([30], dtype=torch.uint32)))
        vals, _ = torch.sort(vals)
        assert del_msh_dlpack.Array1D.torch.has_duplicate_in_sorted_array(vals.cuda())


def test_unique_for_sorted_array():
    idx2val = torch.tensor([1,2,3,5,5,6], dtype=torch.uint32)
    idx2jdx, jdx2val, jdx2idx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(idx2val)
    assert torch.equal(idx2jdx, torch.tensor([0,1,2,3,3,4], dtype=torch.uint32))
    if torch.cuda.is_available():
        d_idx2jdx, d_jdx2val, d_jdx2idx_offset = del_msh_dlpack.Array1D.torch.unique_for_sorted_array(idx2val.cuda())
        assert torch.equal(idx2jdx, d_idx2jdx.cpu())
        assert torch.equal(jdx2idx_offset, d_jdx2idx_offset.cpu())
        assert torch.equal(jdx2val, d_jdx2val.cpu())



