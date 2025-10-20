def binary_radix_tree_and_depth(
    idx2morton,
    num_dim,
    max_depth,
    bnodes,
    bnode2depth,
    stream_ptr=0):
    from ..del_msh_dlpack import quad_oct_tree_binary_radix_tree_and_depth

    quad_oct_tree_binary_radix_tree_and_depth(
        idx2morton, num_dim, max_depth, 
        bnodes, bnode2depth, stream_ptr)




