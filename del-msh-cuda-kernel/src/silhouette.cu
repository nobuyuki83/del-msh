#include <stdio.h>
#include "mat4_col_major.h"
#include "mat2x3_col_major.h"
#include "edge2.h"
#include "dda.h"

struct NeighbourInDirection {
  uint32_t i_pix0;
  cuda::std::array<float,2> c0;
  uint32_t i_pix1;
  cuda::std::array<float,2> c1;
};

class Neighbour {
public:
  const bool is_horizontal;
  cuda::std::array<uint32_t,5> list_i_pix;
  cuda::std::array<cuda::std::array<float,2>,5> list_pos_c;
public:
  __device__
  Neighbour(uint32_t i_pix, uint32_t img_width, bool is_horizontal):
    is_horizontal(is_horizontal)
  {
      const uint32_t iw1 = i_pix % img_width;
      const uint32_t ih1 = i_pix / img_width;
      list_i_pix = {
        ih1 * img_width + iw1,       // c
        ih1 * img_width + iw1 - 1,   // w
        ih1 * img_width + iw1 + 1,   // e
        (ih1 - 1) * img_width + iw1, // s
        (ih1 + 1) * img_width + iw1, // n
      };
      list_pos_c[0] = {float(iw1) + 0.5f, float(ih1) + 0.5f}; // c
      list_pos_c[1] = {float(iw1) - 0.5f, float(ih1) + 0.5f}; // w
      list_pos_c[2] = {float(iw1) + 1.5f, float(ih1) + 0.5f}; // e
      list_pos_c[3] = {float(iw1) + 0.5f, float(ih1) - 0.5f}; // s
      list_pos_c[4] = {float(iw1) + 0.5f, float(ih1) + 1.5f}; // n
  }

  __device__
  auto get(uint32_t i_cnt) const -> NeighbourInDirection {
    if( is_horizontal ){
      uint32_t list_index[4][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}};
      auto idx0 = list_index[i_cnt][0];
      auto idx1 = list_index[i_cnt][1];
      auto i_pix0 = list_i_pix[idx0];
      auto i_pix1 = list_i_pix[idx1];
      auto c0 = list_pos_c[idx0];
      auto c1 = list_pos_c[idx1];
      return NeighbourInDirection{i_pix0, c0, i_pix1, c1};
    }
    else {
      uint32_t list_index[4][2] = {{0, 3}, {3, 0}, {0, 4}, {4, 0}};
      auto idx0 = list_index[i_cnt][0];
      auto idx1 = list_index[i_cnt][1];
      auto i_pix0 = list_i_pix[idx0];
      auto i_pix1 = list_i_pix[idx1];
      auto c0 = list_pos_c[idx0];
      auto c1 = list_pos_c[idx1];
      return NeighbourInDirection{i_pix0, c0, i_pix1, c1};
    }
  }
};


extern "C" {

__global__
void silhouette_fwd(
  const uint32_t num_edge,
  uint32_t *edge2vtx,
  const uint32_t img_w,
  const uint32_t img_h,
  float* pix2occu,
  const uint32_t* pix2tri,
  const float *vtx2xyz,
  const float *transform_world2pix)
{
    int i_edge = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_edge >= num_edge ){ return; }
    //
    const uint32_t i0_vtx = edge2vtx[i_edge*2+0];
    const uint32_t i1_vtx = edge2vtx[i_edge*2+1];
    const float* p0 = vtx2xyz + i0_vtx*3;
    const float* p1 = vtx2xyz + i1_vtx*3;
    const auto q0 = mat4_col_major::transform_homogeneous(transform_world2pix, p0);
    const auto q1 = mat4_col_major::transform_homogeneous(transform_world2pix, p1);
    float v01[2] = {q1[0]-q0[0], q1[1]-q0[1]};
    const bool is_horizontal = abs(v01[0]) < abs(v01[1]);
    auto dda = DDA(q0.data(), q1.data(), img_w, img_h);
    while(true) {
      const auto pixel_coord = dda.pixel();
      auto iw = pixel_coord.ix;
      auto ih = pixel_coord.iy;
      if( iw < 0 || ih < 0 || iw >= img_w || ih >= img_h ){ continue; }
      auto ne = Neighbour(ih * img_w + iw, img_w, is_horizontal);
      for(uint32_t i_cnt=0;i_cnt<4;++i_cnt) {
        auto ni = ne.get(i_cnt);
        if( pix2tri[ni.i_pix0] == UINT32_MAX || pix2tri[ni.i_pix1] != UINT32_MAX ){
            continue;
        }
        auto res = edge2::intersection_edge2(ni.c0.data(), ni.c1.data(), q1.data(), q0.data());
        if( !res ){ continue; }
        const float rc = res.value().r0;
        assert( 0.f<=rc && rc <= 1.f);
        if( rc < 0.5  ){
            pix2occu[ni.i_pix0] = 0.5f + rc;
        } else {
            pix2occu[ni.i_pix1] = rc - 0.5f;
        }
      }
      if( !dda.is_valid() ){ break; } // this is here to allow overrun one pixel to make the line connected.
      dda.move();
    }
}


__global__
void silhouette_bwd(
       uint32_t num_edge,
       const uint32_t *edge2vtx_contour,
       const float *vtx2xyz,
       float *dldw_vtx2xyz,
       uint32_t img_w,
       uint32_t img_h,
       const float *dldw_pix2occl,
       const uint32_t *pix2tri,
       const float* transform_world2pix)
{
    int i_edge = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_edge >= num_edge ){ return; }
    //
    const uint32_t i0_vtx = edge2vtx_contour[i_edge*2+0];
    const uint32_t i1_vtx = edge2vtx_contour[i_edge*2+1];
    const float* p0 = vtx2xyz + i0_vtx*3;
    const float* p1 = vtx2xyz + i1_vtx*3;
    const auto q0 = mat4_col_major::transform_homogeneous(transform_world2pix, p0);
    const auto q1 = mat4_col_major::transform_homogeneous(transform_world2pix, p1);
    float v01[2] = {q1[0]-q0[0], q1[1]-q0[1]};
    const bool is_horizontal = abs(v01[0]) < abs(v01[1]);
    auto dda = DDA(q0.data(), q1.data(), img_w, img_h);
    float dldw_p0[3] = {0.f, 0.f, 0.f};
    float dldw_p1[3] = {0.f, 0.f, 0.f};
    while(true) {
      const auto pixel_coord = dda.pixel();
      auto iw = pixel_coord.ix;
      auto ih = pixel_coord.iy;
      if( iw < 0 || ih < 0 || iw >= img_w || ih >= img_h ){ continue; }
      auto ne = Neighbour(ih * img_w + iw, img_w, is_horizontal);
      for(uint32_t i_cnt=0;i_cnt<4;++i_cnt) {
        const auto ni = ne.get(i_cnt);
        if( pix2tri[ni.i_pix0] == UINT32_MAX || pix2tri[ni.i_pix1] != UINT32_MAX ){
            continue;
        }
        auto res = edge2::intersection_edge2(ni.c0.data(), ni.c1.data(), q1.data(), q0.data());
        if( !res ){ continue; }
        const float rc = res.value().r0;
        assert( 0.f <= rc && rc <= 1.f);
        float dldr0 = ( rc < 0.5f ) ? dldw_pix2occl[ni.i_pix0] : dldw_pix2occl[ni.i_pix1];
        const auto diff =
          edge2::dldw_intersection_edge2(ni.c0.data(), ni.c1.data(), q1.data(), q0.data(), dldr0, 0.f);
        const auto dldq1 = diff.dlds1;
        const auto dldq0 = diff.dlde1;
        const auto dqdp0_3x3 = mat4_col_major::jacobian_transform(transform_world2pix, p0);
        const auto dqdp1_3x3 = mat4_col_major::jacobian_transform(transform_world2pix, p1);
        const auto dqdp0_2x3 = mat3_col_major::to_mat2x3_col_major_xy(dqdp0_3x3.data()); // 2x3 matrix
        const auto dqdp1_2x3 = mat3_col_major::to_mat2x3_col_major_xy(dqdp1_3x3.data()); // 2x3 matrix
        const auto dldp0 = mat2x3_col_major::vec3_from_mult_transpose_vec2(dqdp0_2x3.data(), dldq0.data());
        const auto dldp1 = mat2x3_col_major::vec3_from_mult_transpose_vec2(dqdp1_2x3.data(), dldq1.data());
        vec3::add_in_place(dldw_p0, dldp0.data());
        vec3::add_in_place(dldw_p1, dldp1.data());
      }
      if( !dda.is_valid() ){ break; } // this is here to allow overrun one pixel to make the line connected.
      dda.move();
    }
    atomicAdd(dldw_vtx2xyz + i0_vtx * 3+0, dldw_p0[0]*1.f);
    atomicAdd(dldw_vtx2xyz + i0_vtx * 3+1, dldw_p0[1]*1.f);
    atomicAdd(dldw_vtx2xyz + i0_vtx * 3+2, dldw_p0[2]*1.f);
    atomicAdd(dldw_vtx2xyz + i1_vtx * 3+0, dldw_p1[0]*1.f);
    atomicAdd(dldw_vtx2xyz + i1_vtx * 3+1, dldw_p1[1]*1.f);
    atomicAdd(dldw_vtx2xyz + i1_vtx * 3+2, dldw_p1[2]*1.f);
}


}