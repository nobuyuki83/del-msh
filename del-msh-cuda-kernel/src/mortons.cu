#include "mat4_col_major.h"
#include "mat3_col_major.h"
#include "mortons.h"

extern "C" {

__global__
void vtx2morton(
    uint32_t num_vtx,
    const float* vtx2co,
    uint32_t num_dim,
    const float* transform_co2uni,
    uint32_t* vtx2morton)
{
  const uint32_t i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
  if (i_vtx >= num_vtx) return;
  assert( num_dim == 2 || num_dim == 3 );
  //
  if (num_dim == 2 ) {
    const float* xy0 = vtx2co + i_vtx * 2;
    auto xy1 = mat3_col_major::transform_homogeneous(transform_co2uni, xy0);
    assert(xy1[0]>-1.0e-7f && xy1[0]<1.0f+1.e-7f);
    assert(xy1[1]>-1.0e-7f && xy1[1]<1.0f+1.e-7f);
    uint32_t mc = mortons::device_morton_code2(xy1[0], xy1[1]);
    vtx2morton[i_vtx] = mc;
  }
  else if( num_dim == 3 ) {
    const float* xyz0 = vtx2co + i_vtx * 3;
    auto xyz1 = mat4_col_major::transform_homogeneous(transform_co2uni, xyz0);
    assert(xyz1[0]>-1.0e-7f && xyz1[0]<1.0f+1.e-7f);
    assert(xyz1[1]>-1.0e-7f && xyz1[1]<1.0f+1.e-7f);
    assert(xyz1[2]>-1.0e-7f && xyz1[2]<1.0f+1.e-7f);
    uint32_t mc = mortons::device_morton_code3(xyz1[0], xyz1[1], xyz1[2]);
    vtx2morton[i_vtx] = mc;
  }
}

}