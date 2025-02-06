#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <cfloat>
#include "mat4_col_major.h"
#include "tri3.h"
#include "aabb.h"

__device__
auto ray_for_pixel(
    uint32_t i_pix,
    uint32_t img_w,
    uint32_t img_h,
    const float* transform_ndc2world) -> thrust::pair< cuda::std::array<float,3>, cuda::std::array<float,3>>
{
    using V3f = cuda::std::array<float,3>;
    const uint32_t i_h = i_pix / img_w;
    const uint32_t i_w = i_pix - i_h * img_w;
    const float x0 = 2.f * (float(i_w) + 0.5f) / float(img_w) - 1.f; // ndc_x
    const float y0 = 1.f - 2.f * (float(i_h) + 0.5f) / float(img_h); // ndc_y
    const auto q0 = mat4_col_major::transform_homogeneous(transform_ndc2world, V3f{x0, y0, +1.f}.data());
    const auto q1 = mat4_col_major::transform_homogeneous(transform_ndc2world, V3f{x0, y0, -1.f}.data());
    const auto v01 = vec3::sub(q1.data(), q0.data());
    return thrust::make_pair(q0,v01);
}

extern "C" {
__global__
void pix_to_tri(
  uint32_t *pix2tri,
  const uint32_t num_tri,
  const uint32_t *tri2vtx,
  const float *vtx2xyz,
  const uint32_t img_w,
  const uint32_t img_h,
  const float *transform_ndc2world,
  const uint32_t *bvhnodes,
  const float *aabbs)
{
    int i_pix = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_pix >= img_w * img_h ){ return; }
    //
    auto ray = ray_for_pixel(i_pix, img_w, img_h, transform_ndc2world);
    /*
    for(int i_tri=0;i_tri<num_tri;++i_tri){
        const float* p0 = vtx2xyz + tri2vtx[i_tri*3+0]*3;
        const float* p1 = vtx2xyz + tri2vtx[i_tri*3+1]*3;
        const float* p2 = vtx2xyz + tri2vtx[i_tri*3+2]*3;
        const auto res = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
        if(!res){ continue; }
        pix2tri[i_pix] = i_tri;
        return;
    }
    return;
    */
    constexpr int STACK_SIZE = 128;
    uint32_t stack[STACK_SIZE];
    float hit_depth = FLT_MAX;
    uint32_t hit_idxtri = UINT32_MAX;
    int32_t i_stack = 1;
    stack[0] = 0;
    while( i_stack > 0 ){
        uint32_t i_bvhnode = stack[i_stack-1];
        --i_stack;
        if( !aabb::is_intersect_ray<3>(aabbs + i_bvhnode*6, ray.first.data(), ray.second.data() ) ){
              continue;
        }
        if( bvhnodes[i_bvhnode * 3 + 2] == UINT32_MAX ){
            const uint32_t i_tri = bvhnodes[i_bvhnode * 3 + 1];
            const float* p0 = vtx2xyz + tri2vtx[i_tri*3+0]*3;
            const float* p1 = vtx2xyz + tri2vtx[i_tri*3+1]*3;
            const float* p2 = vtx2xyz + tri2vtx[i_tri*3+2]*3;
            const auto r = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
            if(r) { // hit triangle
                float depth = r.value();
                if( depth < hit_depth ){
                    hit_depth = depth;
                    hit_idxtri = i_tri;
                }
            }
            continue;
        }
        stack[i_stack] = bvhnodes[i_bvhnode * 3 + 1];
        ++i_stack;
        stack[i_stack] = bvhnodes[i_bvhnode * 3 + 2];
        ++i_stack;
    }
    pix2tri[i_pix] = hit_idxtri;
}


}