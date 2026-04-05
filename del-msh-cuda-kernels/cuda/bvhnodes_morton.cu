#include "mortons.h"

extern "C" {

__global__
void tri2cntr(
  float* tri2cntr,
  uint32_t num_tri,
  const uint32_t* tri2vtx,
  const float* vtx2xyz)
{
  const unsigned int i_tri = blockDim.x * blockIdx.x + threadIdx.x;
  if (i_tri >= num_tri) return;
  //
  const uint32_t* node2vtx = tri2vtx + i_tri*3;
  const float ratio = 1.0/3.0;
  for(int i_dim=0;i_dim<3;++i_dim){
    float sum = 0.;
    for(uint32_t i_node=0;i_node<3;++i_node){
        sum += vtx2xyz[node2vtx[i_node]*3+i_dim];
    }
    tri2cntr[i_tri*3+i_dim] = sum * ratio;
  }
}



__global__
void kernel_MortonCode_BVHTopology(
    const uint32_t num_idx,
    uint32_t* bvhnodes,
    const uint32_t *idx2morton,
    const uint32_t *idx2obj)
{
  const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= num_idx-1) return;
  const unsigned int ini = idx;
  const unsigned int nni = num_idx-1;
  // -------------------------------
  const int2 range = mortons::device_MortonCode_DeterminRange(idx2morton,num_idx,ini);
  const int isplit = mortons::device_MortonCode_FindSplit(idx2morton,range.x,range.y);
  // printf("%d --> %d %d  %d\n",ini, range.x, range.y, isplit);
  // -------------------------------
  if( range.x == isplit ){
    const unsigned int inlA = nni+isplit;
    bvhnodes[ini*3+1] = inlA;
    bvhnodes[inlA*3] = ini;
    bvhnodes[inlA*3+1] = idx2obj[isplit];
    bvhnodes[inlA*3+2] = UINT_MAX;
  }
  else{
    const unsigned int iniA = isplit;
    bvhnodes[ini*3+1] = iniA;
    bvhnodes[iniA*3] = ini;
  }
  // ----
  if( range.y == isplit+1 ){
    const unsigned int inlB = nni+isplit+1;
    bvhnodes[ini*3+2] = inlB;
    bvhnodes[inlB*3] = ini;
    bvhnodes[inlB*3+1] = idx2obj[isplit+1];
    bvhnodes[inlB*3+2] = UINT_MAX;
  }
  else{
    const unsigned int iniB = isplit+1;
    bvhnodes[ini*3+2] = iniB;
    bvhnodes[iniB*3] = ini;
  }
  if (idx == 0) {
    bvhnodes[0] = UINT_MAX;
  }
}


}


