def bnodes_and_bnode2depth_and_bnode2onode(
    idx2morton,
    num_dim,
    max_depth,
    bnodes,
    bnode2depth,
    bnode2onode,
    stream_ptr=0):
    from ..del_msh_dlpack import quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode

    quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode(
        idx2morton, num_dim, max_depth, 
        bnodes, bnode2depth, bnode2onode,
        stream_ptr)





