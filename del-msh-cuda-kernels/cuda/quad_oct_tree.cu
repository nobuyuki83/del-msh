#include "mortons.h"

__device__
void morton2center(
    const uint32_t morton,
    const uint32_t num_dim,
    const uint32_t depth,
    const uint32_t max_depth,
    float* center)
{
    uint32_t key = morton >> ((max_depth - depth) * num_dim);
    for(int i_dim=0;i_dim<num_dim;++i_dim) {
        center[i_dim] = 0.5;
    }
    for(int i_depth=0;i_depth<depth;++i_depth) {
        for(int i_dim=0;i_dim<num_dim;++i_dim) {
            int j = num_dim - i_dim - 1;
            center[j] += static_cast<float>(key & 1);
            center[j] *= 0.5f;
            key >>= 1;
        }
    }
}

extern "C" {

__global__
void binary_radix_tree_and_depth(
  uint32_t num_bnode,
  uint32_t* idx2morton,
  uint32_t num_dim,
  uint32_t max_depth,
  uint32_t* bnodes,
  uint32_t* bnode2depth)
{
    const unsigned int i_bnode = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_bnode >= num_bnode ) return;
    if( i_bnode == 0 ){
      bnodes[0] = UINT32_MAX;
    }
    uint32_t num_idx = num_bnode + 1;
//   num_branch
    int2 range = mortons::device_MortonCode_DeterminRange(idx2morton, num_idx, i_bnode);
    int i_split = mortons::device_MortonCode_FindSplit(idx2morton, range.x, range.y);
    assert(i_split != UINT32_MAX);
    if( range.x == i_split ){
        int i_bnode_a = num_bnode + i_split; // leaf node
        bnodes[i_bnode * 3 + 1] = i_bnode_a;
    } else {
        int i_bnode_a = i_split;
        bnodes[i_bnode * 3 + 1] = i_bnode_a;
        bnodes[i_bnode_a * 3] = i_bnode;
    }
    // ----
    if( range.y == i_split + 1 ){
        int i_bnode_b = num_bnode + i_split + 1;
        bnodes[i_bnode * 3 + 2] = i_bnode_b;
    } else {
        int i_bnode_b = i_split + 1;
        bnodes[i_bnode * 3 + 2] = i_bnode_b;
        bnodes[i_bnode_b * 3] = i_bnode;
    }
    // ---
    int delta_split = mortons::device_Delta(i_split, i_split+1, idx2morton, num_idx);
    // 3D: morton code use 3x10 bits. 32 - 3x10 = 2. Leading two bits are always zero.
    int offset = 32 - num_dim * max_depth;
    bnode2depth[i_bnode] = (delta_split - offset) / num_dim;
}


__global__
void bnode2isonode_and_idx2bnode(
    uint32_t num_bnode,
    const uint32_t* bnodes,
    const uint32_t* bnode2depth,
    uint32_t* bnode2isonode,
    uint32_t* idx2bnode)
{
    const unsigned int i_bnode = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_bnode >= num_bnode ) return;
    // make idx2bnode
    if( bnodes[i_bnode * 3 + 1] >= num_bnode ){
        uint32_t idx = bnodes[i_bnode * 3 + 1] - num_bnode;
        assert(idx<num_bnode+1);
        idx2bnode[idx] = i_bnode;
    }
    if( bnodes[i_bnode * 3 + 2] >= num_bnode ){
        uint32_t idx = bnodes[i_bnode * 3 + 2] - num_bnode;
        assert(idx<num_bnode+1);
        idx2bnode[idx] = i_bnode;
    }
    //
    if( i_bnode == 0 ){
        return;
    }
    const uint32_t i_bnode_parent = bnodes[i_bnode * 3];
    if( bnode2depth[i_bnode] != bnode2depth[i_bnode_parent] ){
        bnode2isonode[i_bnode-1] = 1; // shift for exclusive scan
    }
    else {
        bnode2isonode[i_bnode-1] = 0;
    }
}


__global__
void make_tree_from_binary_radix_tree(
    const uint32_t num_vtx,
    const uint32_t* bnodes,
    const uint32_t* bnode2onode,
    const uint32_t* bnode2depth,
    const uint32_t* idx2bnode,
    const uint32_t* idx2morton,
    const uint32_t num_onode,
    const uint32_t max_depth,
    uint32_t num_dim,
    uint32_t* onodes,
    uint32_t* onode2depth,
    float* onode2center,
    uint32_t* idx2onode,
    float* idx2center)
{
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_vtx ) return;
    //
    const int num_child = 1 << num_dim; // 8 for 3D
    const int nlink = num_child + 1; // 9 for 3D, 5 for 2D
    {
        // set leaf
        int idx = i;
        uint32_t i_onode_parent = UINT32_MAX;
        uint32_t depth_parent = UINT32_MAX;
        {
            uint32_t i_bnode_cur = idx2bnode[idx];
            while(1) {
                if( i_bnode_cur == 0 || bnode2onode[i_bnode_cur] != bnode2onode[i_bnode_cur - 1] )
                {
                    i_onode_parent = bnode2onode[i_bnode_cur];
                    depth_parent = bnode2depth[i_bnode_cur];
                    break;
                }
                i_bnode_cur = bnodes[i_bnode_cur * 3];
            }
        };
        assert(i_onode_parent < num_onode );
        assert(depth_parent < max_depth);
        const uint32_t key = idx2morton[idx];
        const int i_child =
             (key >> ((max_depth - 1 - depth_parent) * num_dim)) & (num_child - 1);
        // println!("leaf: {} {} {}", i_onode_parent, i_child, depth_parent);
        const uint32_t idx_onodes = i_onode_parent * nlink + 1 + i_child;
        // assert(onodes[idx_onodes] == UINT32_MAX);
        //
        onodes[idx_onodes] = idx + num_onode;
        idx2onode[idx] = i_onode_parent;
        float center[3];
        morton2center(idx2morton[idx], num_dim, max_depth, max_depth, center);
        for (int i_dim=0;i_dim<num_dim;++i_dim) {
            idx2center[idx * num_dim + i_dim] = center[i_dim];
        }
    }
    if( i == num_vtx - 1 ) { return; }
    {
        int i_bnode = i;
        if( i_bnode == 0 ){
            for(int i_dim=0;i_dim<num_dim;++i_dim){
                onode2center[i_dim] = 0.5;
            }
            return;
        }
        if( bnode2onode[i_bnode - 1] == bnode2onode[i_bnode] ){
            return;
        }
        uint32_t i_onode_parent = UINT32_MAX;
        uint32_t depth_parent = UINT32_MAX;
        {
            uint32_t i_bnode_parent = bnodes[i_bnode * 3];
            while(1) {
                if( i_bnode_parent == 0
                    || bnode2onode[i_bnode_parent] != bnode2onode[i_bnode_parent - 1])
                {
                    i_onode_parent = bnode2onode[i_bnode_parent];
                    depth_parent = bnode2depth[i_bnode_parent];
                    break;
                }
                i_bnode_parent = bnodes[i_bnode_parent * 3];
            }
        };
        assert(depth_parent < max_depth);
        assert(i_onode_parent < num_onode );
        const uint32_t morton = idx2morton[i_bnode];
        const int i_child = (morton >> ((max_depth - 1 - depth_parent) * num_dim)) & (num_child - 1);
        const uint32_t idx_onodes = i_onode_parent * nlink + 1 + i_child;
        // assert( onodes[idx_onodes] == UINT32_MAX );
        //
        int i_onode = bnode2onode[i_bnode];
        onodes[idx_onodes] = i_onode;
        onodes[i_onode * nlink] = i_onode_parent;
        //
        uint32_t depth = bnode2depth[i_bnode];
        float center[3];
        morton2center(morton, num_dim, depth, max_depth, center);
        for(int i_dim = 0; i_dim<num_dim;++i_dim) {
            onode2center[i_onode * num_dim + i_dim] = center[i_dim];
        }
        onode2depth[i_onode] = depth;
    }
}

__global__
void aggregate(
    uint32_t num_idx,
    uint32_t num_dim,
    const float* idx2val,
    const uint32_t* idx2onode,
    uint32_t num_link,
    const uint32_t* onodes,
    float* onode2aggval)
{
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_idx ) return;
    // ---------------
    const float* val = idx2val + idx * num_dim;
    uint32_t i_onode = idx2onode[idx];
    while(i_onode != UINT32_MAX){
        for(int i_dim=0; i_dim<num_dim; ++i_dim ){
           atomicAdd(onode2aggval + i_onode*num_dim + i_dim, val[i_dim]);
        }
        i_onode = onodes[i_onode*num_link];
    }
}


}