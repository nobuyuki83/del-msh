#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include "tri3.h"
#include "vec3.h"
// #include "mat3_col_major.h"
#include "mat4_col_major.h"
#include "ray_for_pixel.h"

extern "C" {

__global__
void fwd_pix2depth(
  float* pix2depth,
  const uint32_t *pix2tri,
  const uint32_t num_tri,
  const uint32_t *tri2vtx,
  const float *vtx2xyz,
  const uint32_t img_w,
  const uint32_t img_h,
  const float *transform_ndc2world,
  const float *transform_world2ndc
  )
  {
    int i_pix = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_pix >= img_w * img_h ){ return; }
    //
    auto ray = ray_for_pixel(i_pix, img_w, img_h, transform_ndc2world);
    const uint32_t i_tri = pix2tri[i_pix];
    // printf("%d", i_tri);
    if( i_tri == UINT32_MAX ){
        pix2depth[i_pix] = 0.f;
        return;
    }
    assert( i_tri < num_tri );
    const uint32_t iv0 = tri2vtx[i_tri*3+0];
    const uint32_t iv1 = tri2vtx[i_tri*3+1];
    const uint32_t iv2 = tri2vtx[i_tri*3+2];
    // printf("%d %d %d", iv0, iv1, iv2);
    const float* p0 = vtx2xyz + iv0*3;
    const float* p1 = vtx2xyz + iv1*3;
    const float* p2 = vtx2xyz + iv2*3;
    auto opt_depth = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
    if(!opt_depth){ return; }
    float depth = opt_depth.value();
    const auto pos_world = vec3::axpy(depth, ray.second.data(), ray.first.data());
    const auto pos_ndc = mat4_col_major::transform_homogeneous(transform_world2ndc, pos_world.data());
    const float depth_ndc = (pos_ndc[2] + 1.f) * 0.5f;
    pix2depth[i_pix] = depth_ndc;
  }

  __global__
  void bwd_wrt_vtx2xyz(
    float* dw_vtx2xyz,
    uint32_t img_w,
    uint32_t img_h,
    const uint32_t* pix2tri,
    uint32_t num_tri,
    const uint32_t* tri2vtx,
    const float* vtx2xyz,
    const float* dw_pix2depth,
    const float* transform_ndc2world,
    const float* transform_world2ndc
  )
  {
      int i_pix = blockDim.x * blockIdx.x + threadIdx.x;
      if( i_pix >= img_w * img_h ){ return; }
      //
      const uint32_t i_tri = pix2tri[i_pix];
      if( i_tri == UINT32_MAX ){ return; }
      auto ray = ray_for_pixel(i_pix, img_w, img_h, transform_ndc2world);
      const uint32_t iv[3] = { tri2vtx[i_tri*3+0], tri2vtx[i_tri*3+1], tri2vtx[i_tri*3+2] };
      const float* p0 = vtx2xyz + iv[0]*3;
      const float* p1 = vtx2xyz + iv[1]*3;
      const float* p2 = vtx2xyz + iv[2]*3;
      const float dw_depth = dw_pix2depth[i_pix];
      float mag;
      {
        auto opt_depth = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
        assert(opt_depth);
        float depth = opt_depth.value();
        const auto pos_world = vec3::axpy(depth, ray.second.data(), ray.first.data());
        const auto j_trans = mat4_col_major::jacobian_transform(transform_world2ndc, pos_world.data());
        const auto jtrans_dir = mat3_col_major::mult_vec(j_trans.data(), ray.second.data());
        mag = dw_depth * jtrans_dir[2] * 0.5;
      }
      const auto opt_res = tri3::intersection_against_line_bwd_wrt_tri(
        p0, p1, p2,
        ray.first.data(), ray.second.data(),
        mag, 0.f, 0.f);
      const auto res = opt_res.value();
      const cuda::std::array<float,3> dp[3] = {res.d_p0, res.d_p1, res.d_p2};
      for(int ino=0;ino<3;++ino){
        for(int idim=0;idim<3;++idim){
            float* po = dw_vtx2xyz + iv[ino] * 3 + idim;
            float pi = dp[ino][idim];
            atomicAdd(po, pi);
        }
      }
  }
}


