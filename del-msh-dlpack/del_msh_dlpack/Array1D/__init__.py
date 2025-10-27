def permute(old2val, new2old, new2val, stream_ptr=0):
    from ..del_msh_dlpack import array1d_permute

    array1d_permute(old2val, new2old, new2val, stream_ptr)


def argsort(idx2val, jdx2idx, stream_ptr=0):
    from ..del_msh_dlpack import array1d_argsort

    array1d_argsort(idx2val, jdx2idx, stream_ptr)


def has_duplicate_sorted_array(idx2val, stream_ptr=0) -> bool:
    from ..del_msh_dlpack import array1d_has_duplicate_sorted_array

    return array1d_has_duplicate_sorted_array(idx2val, stream_ptr)


def unique_for_sorted_array(idx2val, idx2jdx, stream_ptr=0):
    from ..del_msh_dlpack import array1d_unique_for_sorted_array

    return array1d_unique_for_sorted_array(idx2val, idx2jdx, stream_ptr)


def unique_jdx2val_jdx2idx(idx2val, idx2jdx, jdx2val, jdx2idx, stream_ptr=0):
    from ..del_msh_dlpack import array1d_unique_jdx2val_jdx2idx

    array1d_unique_jdx2val_jdx2idx(idx2val, idx2jdx, jdx2val, jdx2idx, stream_ptr)