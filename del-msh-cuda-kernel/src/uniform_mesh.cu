#include <cstdint> // for uint32_t
#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {

__global__
void vtx2valence(
    const uint32_t num_elem,
    const int32_t *elem2vtx,
    const int32_t num_node,
    int32_t* vtx2valence)
{
    int i_elem = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_elem >= num_elem) { return; }
    //
    for(int i_node=0;i_node<num_node;++i_node){
        uint32_t i0_vtx = elem2vtx[i_elem*num_node+i_node];
        atomicAdd(vtx2valence+i0_vtx, 1);
    }
}

__global__
void fill_idx2vtx(
    const uint32_t num_elem,
    const int32_t *elem2vtx,
    const int32_t num_node,
    uint32_t* vtx2idx,
    uint32_t* idx2vtx)
{
    int i_elem = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_elem >= num_elem) { return; }
    //
    for(int i_node=0;i_node<num_node;++i_node){
        uint32_t i0_vtx = elem2vtx[i_elem*num_node+i_node];
        uint32_t i0_idx = atomicAdd(vtx2idx+i0_vtx, 1);
        idx2vtx[i0_idx] = i_elem;
    }
}


}
