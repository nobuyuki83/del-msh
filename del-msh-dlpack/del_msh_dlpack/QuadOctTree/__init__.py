def bnodes_and_bnode2depth_and_bnode2onode(
    idx2morton,
    num_dim,
    max_depth,
    bnodes,
    bnode2depth,
    bnode2onode,
    idx2bnode,
    stream_ptr=0):
    from ..del_msh_dlpack import quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode

    quad_oct_tree_bnodes_and_bnode2depth_and_bnode2onode_and_idx2bnode(
        idx2morton, num_dim, max_depth, 
        bnodes, bnode2depth, bnode2onode, idx2bnode,
        stream_ptr)



def make_tree_from_binary_radix_tree(
    bnodes,
    bnode2onode,
    bnode2depth,
    idx2bnode,
    idx2morton,
    num_dim: int,
    max_depth: int,
    onodes,
    onode2depth,
    onode2center,
    idx2onode,
    idx2center,
    stream_ptr=0):
    from ..del_msh_dlpack import quad_oct_tree_make_tree_from_binary_radix_tree

    quad_oct_tree_make_tree_from_binary_radix_tree(
        bnodes, bnode2onode, bnode2depth,
        idx2bnode, idx2morton,
        num_dim, max_depth,
        onodes, onode2depth, onode2center, idx2onode, idx2center,
        stream_ptr)

