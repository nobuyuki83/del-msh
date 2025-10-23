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



def make_tree_from_binary_radix_tree(
    bnodes,
    bnode2onode,
    bnode2depth,
    idx2morton,
    num_dim: int,
    max_depth: int,
    onodes,
    onode2depth,
    onode2center,
    idx2onode,
    idx2center):
    from ..del_msh_dlpack import quad_oct_tree_make_tree_from_binary_radix_tree

    quad_oct_tree_make_tree_from_binary_radix_tree(
        bnodes, bnode2onode, bnode2depth, idx2morton,
        num_dim, max_depth,
        onodes, onode2depth, onode2center, idx2onode, idx2center)

