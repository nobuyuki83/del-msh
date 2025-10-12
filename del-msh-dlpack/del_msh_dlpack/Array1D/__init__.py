def permute(old2val, new2old, new2val, stream_ptr=0):
    from ..del_msh_dlpack import array1d_permute

    array1d_permute(old2val, new2old, new2val, stream_ptr)


def argsort(idx2val, jdx2idx, stream_ptr=0):
    from ..del_msh_dlpack import array1d_argsort

    array1d_argsort(idx2val, jdx2idx, stream_ptr)
