import torch
from .. import util_torch

def permute(old2val: torch.Tensor, new2old: torch.Tensor):
    """Permute a 1D array according to an index mapping.

    Args:
        old2val: (n,) uint32 - source array
        new2old: (n,) uint32 - mapping from new indices to old indices
    Returns:
        new2val: (n,) uint32 - permuted array where new2val[i] = old2val[new2old[i]]
    """
    n = old2val.shape[0]
    device = old2val.device
    #
    util_torch.assert_shape_dtype_device(old2val, shape=(n, ), dtype=torch.uint32, device=device)
    util_torch.assert_shape_dtype_device(new2old, shape=(n, ), dtype=torch.uint32, device=device)
    #
    new2val = torch.empty(size=(n,), device=device, dtype=torch.uint32)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Array1D

    Array1D.permute(
        util_torch.to_dlpack_safe(old2val, stream_ptr),
        util_torch.to_dlpack_safe(new2old, stream_ptr),
        util_torch.to_dlpack_safe(new2val, stream_ptr),
        stream_ptr,
    )
    return new2val


def argsort(idx2val: torch.Tensor):
    """Sort a 1D array and return both the sorted indices and sorted values.

    Args:
        idx2val: (n,) uint32 - array to sort
    Returns:
        jdx2idx: (n,) uint32 - original indices in sorted order (argsort result)
        jdx2val: (n,) uint32 - sorted values
    """
    n = idx2val.shape[0]
    device = idx2val.device
    #
    assert idx2val.shape == (n,)
    assert idx2val.dtype == torch.uint32
    #
    jdx2idx = torch.empty(size=(n,), device=device, dtype=torch.uint32)
    jdx2val = idx2val.clone()
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Array1D

    Array1D.argsort(
        util_torch.to_dlpack_safe(jdx2val, stream_ptr),
        util_torch.to_dlpack_safe(jdx2idx, stream_ptr),
        stream_ptr,
    )
    return jdx2idx, jdx2val

#################################
# below sorted

def has_duplicate_in_sorted_array(idx2val: torch.Tensor):
    """Check whether a sorted array contains any duplicate values.

    Args:
        idx2val: (n,) uint32 - sorted array to check
    Returns:
        bool - True if any duplicate values exist
    """
    n = idx2val.shape[0]
    device = idx2val.device
    #
    assert idx2val.shape == (n,)
    assert idx2val.dtype == torch.uint32
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    from .. import Array1D

    res = Array1D.has_duplicate_sorted_array(
        util_torch.to_dlpack_safe(idx2val, stream_ptr),
        stream_ptr
    )

    return res


def unique_for_sorted_array(idx2val: torch.Tensor):
    """Compute unique values and their groupings from a sorted array.

    Args:
        idx2val: (num_idx,) uint32 - sorted array (must be sorted)
    Returns:
        idx2jdx: (num_idx,) uint32 - maps each element to its unique group index
        jdx2val: (num_jdx,) uint32 - unique values
        jdx2idx_offset: (num_jdx+1,) uint32 - offset array into idx2val per unique group
    """
    from .. import Array1D
    #
    num_idx = idx2val.shape[0]
    device = idx2val.device
    #
    assert idx2val.shape == (num_idx,)
    assert idx2val.dtype == torch.uint32
    #
    idx2jdx = torch.empty((num_idx,), dtype=torch.uint32, device=device)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    Array1D.unique_for_sorted_array(
        util_torch.to_dlpack_safe(idx2val, stream_ptr),
        util_torch.to_dlpack_safe(idx2jdx, stream_ptr),
        stream_ptr
    )
    #
    num_jdx = idx2jdx[-1].item() + 1
    jdx2val = torch.empty((num_jdx,), dtype=torch.uint32, device=device)
    jdx2idx_offset = torch.empty((num_jdx+1,), dtype=torch.uint32, device=device)
    #
    Array1D.unique_jdx2val_jdx2idx(
        util_torch.to_dlpack_safe(idx2val, stream_ptr),
        util_torch.to_dlpack_safe(idx2jdx, stream_ptr),
        util_torch.to_dlpack_safe(jdx2val, stream_ptr),
        util_torch.to_dlpack_safe(jdx2idx_offset, stream_ptr),
        stream_ptr,
    )
    return idx2jdx, jdx2val, jdx2idx_offset