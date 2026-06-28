#include "del_geo/aabb3.h"


extern "C" {

__global__
void from_trimesh3(
    float* bvhnode2aabb,
    int* bvhbranch2flag,
    unsigned int num_bvhnode,
    const uint32_t* bvhnodes,
    unsigned int num_tri,
    const uint32_t* tri2vtx,
    const float* vtx2xyz,
    float eps)
{
  const unsigned int ii_tri = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii_tri >= num_tri) return;
  //
  unsigned int i_bvhnode = num_tri - 1 + ii_tri;
  { // make aabb for triangle
    assert( bvhnodes[i_bvhnode*3+2] == UINT_MAX );
    assert( bvhnodes[i_bvhnode*3] < num_tri-1 );
    const int i_tri = bvhnodes[i_bvhnode*3+1];
    assert(i_tri >= 0 && i_tri < num_tri);
    const unsigned int i0 = tri2vtx[i_tri * 3 + 0];
    const unsigned int i1 = tri2vtx[i_tri * 3 + 1];
    const unsigned int i2 = tri2vtx[i_tri * 3 + 2];
    const float *p0 = vtx2xyz + i0 * 3;
    const float *p1 = vtx2xyz + i1 * 3;
    const float *p2 = vtx2xyz + i2 * 3;
    float* aabb = bvhnode2aabb + i_bvhnode*6;
    aabb3::set_point(aabb, p0,eps);
    aabb3::add_point(aabb, p1,eps);
    aabb3::add_point(aabb, p2,eps);
  }
  //----------------------------------------------------

  i_bvhnode = bvhnodes[i_bvhnode*3];
  while(true){
    assert( i_bvhnode < num_tri-1 );
    //assert( dNodeBVH[ino0].ichild[0] >= 0 );
    //assert( dNodeBVH[ino0].ichild[1] >= 0 );
    const unsigned int i_bvhnode_c0 = bvhnodes[i_bvhnode*3+1];
    const unsigned int i_bvhnode_c1 = bvhnodes[i_bvhnode*3+2];
    assert( bvhnodes[i_bvhnode_c0*3] == i_bvhnode );
    assert( bvhnodes[i_bvhnode_c1*3] == i_bvhnode );
    assert( i_bvhnode_c0 < num_tri*2-1 );
    assert( i_bvhnode_c1 < num_tri*2-1 );
    const int iflg_old = atomicCAS(bvhbranch2flag+i_bvhnode,0,1);
    if( iflg_old == 0 ){ // let the another branch of the binary tree do the work
      return;
    }
    __threadfence(); // sync global memory
    // ---------------------------------------
    aabb3::set_merged_two_aabbs(
        bvhnode2aabb+i_bvhnode*6,
        bvhnode2aabb+i_bvhnode_c0*6,
        bvhnode2aabb+i_bvhnode_c1*6);
    // ----------------------------------------
    if( bvhnodes[i_bvhnode*3] == UINT_MAX ){ assert(i_bvhnode==0); return; }
    i_bvhnode = bvhnodes[i_bvhnode*3];
  }

}

__global__
void from_polyhedron_mesh(
    float* bvhnode2aabb,
    int* bvhbranch2flag,
    unsigned int num_bvhnode,
    const uint32_t* bvhnodes,
    unsigned int num_elem,
    const uint32_t* elem2idx_offset,
    const uint32_t* idx2vtx,
    const float* vtx2xyz,
    float eps)
{
  const unsigned int ii_elem = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii_elem >= num_elem) return;

  unsigned int i_bvhnode = num_elem - 1 + ii_elem;
  { // make aabb for the polygon element
    assert( bvhnodes[i_bvhnode*3+2] == UINT_MAX );
    assert( bvhnodes[i_bvhnode*3] < num_elem-1 );
    const uint32_t i_elem = bvhnodes[i_bvhnode*3+1];
    assert(i_elem < num_elem);
    float* aabb = bvhnode2aabb + i_bvhnode*6;
    const uint32_t i_idx_begin = elem2idx_offset[i_elem];
    const uint32_t i_idx_end   = elem2idx_offset[i_elem + 1];
    const float* p0 = vtx2xyz + idx2vtx[i_idx_begin] * 3;
    aabb3::set_point(aabb, p0, eps);
    for (uint32_t i_idx = i_idx_begin + 1; i_idx < i_idx_end; ++i_idx) {
      const float* p = vtx2xyz + idx2vtx[i_idx] * 3;
      aabb3::add_point(aabb, p, eps);
    }
  }
  //----------------------------------------------------

  i_bvhnode = bvhnodes[i_bvhnode*3];
  while(true){
    assert( i_bvhnode < num_elem-1 );
    const unsigned int i_bvhnode_c0 = bvhnodes[i_bvhnode*3+1];
    const unsigned int i_bvhnode_c1 = bvhnodes[i_bvhnode*3+2];
    assert( bvhnodes[i_bvhnode_c0*3] == i_bvhnode );
    assert( bvhnodes[i_bvhnode_c1*3] == i_bvhnode );
    assert( i_bvhnode_c0 < num_elem*2-1 );
    assert( i_bvhnode_c1 < num_elem*2-1 );
    const int iflg_old = atomicCAS(bvhbranch2flag+i_bvhnode,0,1);
    if( iflg_old == 0 ){ // let the other branch finish first
      return;
    }
    __threadfence(); // sync global memory
    aabb3::set_merged_two_aabbs(
        bvhnode2aabb+i_bvhnode*6,
        bvhnode2aabb+i_bvhnode_c0*6,
        bvhnode2aabb+i_bvhnode_c1*6);
    if( bvhnodes[i_bvhnode*3] == UINT_MAX ){ assert(i_bvhnode==0); return; }
    i_bvhnode = bvhnodes[i_bvhnode*3];
  }

}

}