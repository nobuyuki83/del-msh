#include "mortons.h"

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

}