
def aggregate(
    idx2jdx_offset,
    jdx2kdx,
    kdx2val,
    kdx2aggval,
    stream_ptr=0):

    from ..del_msh_dlpack import offset_array_aggregate
    offset_array_aggregate(idx2jdx_offset, jdx2kdx, kdx2val, kdx2aggval, stream_ptr)