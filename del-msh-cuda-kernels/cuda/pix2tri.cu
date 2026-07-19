#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <cfloat>
#include "del_geo/mat4_col_major.h"
#include "del_geo/tri3.h"
#include "del_geo/aabb.h"
#include "ray_for_pixel.h"

extern "C"{

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
    //
    /*
    pix2tri[i_pix] = UINT32_MAX;
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
            const auto opt_raycoeff_bc = tri3::intersection_against_ray(p0, p1, p2, ray.first.data(), ray.second.data());
            if(opt_raycoeff_bc) { // hit triangle
                float depth = opt_raycoeff_bc.value().ray_coeff;
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

__global__
void interpolate(
    const uint32_t *pix2tri,
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    const float *vtx2val,
    const uint32_t num_vdim,
    const float *transform_ndc2world,
    const uint32_t img_w,
    const uint32_t img_h,
    float *pix2val)
{
    const uint32_t i_pix = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_pix >= img_w * img_h) { return; }
    //
    const uint32_t i_tri = pix2tri[i_pix];
    if (i_tri == UINT32_MAX) {
        for (uint32_t i = 0; i < num_vdim; ++i) {
            pix2val[i_pix * num_vdim + i] = 0.f;
        }
        return;
    }
    //
    auto ray = ray_for_pixel(i_pix, img_w, img_h, transform_ndc2world);
    //
    const uint32_t i0 = tri2vtx[i_tri * 3 + 0];
    const uint32_t i1 = tri2vtx[i_tri * 3 + 1];
    const uint32_t i2 = tri2vtx[i_tri * 3 + 2];
    const float *p0 = vtx2xyz + i0 * 3;
    const float *p1 = vtx2xyz + i1 * 3;
    const float *p2 = vtx2xyz + i2 * 3;
    const auto raycoeff_bc = tri3::intersection_plane_of_tri3_against_line(p0, p1, p2, ray.first.data(), ray.second.data());
    const auto bc = raycoeff_bc.barycentric_coord;
    //
    for (uint32_t i = 0; i < num_vdim; ++i) {
        pix2val[i_pix * num_vdim + i] =
            bc[0] * vtx2val[i0 * num_vdim + i]
          + bc[1] * vtx2val[i1 * num_vdim + i]
          + bc[2] * vtx2val[i2 * num_vdim + i];
    }
}

__global__
void interpolate_bwd(
    const uint32_t *pix2tri,
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    const float *vtx2val,
    const uint32_t num_vdim,
    const float *transform_ndc2world,
    const float *dldw_pix2val,
    const uint32_t img_w,
    const uint32_t img_h,
    float *dldw_vtx2xyz,
    float *dldw_vtx2val)
{
    const uint32_t i_pix = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_pix >= img_w * img_h) { return; }
    //
    const uint32_t i_tri = pix2tri[i_pix];
    if (i_tri == UINT32_MAX) { return; }
    //
    auto ray = ray_for_pixel(i_pix, img_w, img_h, transform_ndc2world);
    //
    const uint32_t i0 = tri2vtx[i_tri * 3 + 0];
    const uint32_t i1 = tri2vtx[i_tri * 3 + 1];
    const uint32_t i2 = tri2vtx[i_tri * 3 + 2];
    const float *p0 = vtx2xyz + i0 * 3;
    const float *p1 = vtx2xyz + i1 * 3;
    const float *p2 = vtx2xyz + i2 * 3;
    // gradient of loss w.r.t. barycentric coords
    float dldw_bc0 = 0.f, dldw_bc1 = 0.f, dldw_bc2 = 0.f;
    for (uint32_t i = 0; i < num_vdim; ++i) {
        const float dl = dldw_pix2val[i_pix * num_vdim + i];
        dldw_bc0 += vtx2val[i0 * num_vdim + i] * dl;
        dldw_bc1 += vtx2val[i1 * num_vdim + i] * dl;
        dldw_bc2 += vtx2val[i2 * num_vdim + i] * dl;
    }
    // bc0 = 1 - bc1 - bc2, so d_bc1 and d_bc2 absorb d_bc0
    dldw_bc1 -= dldw_bc0;
    dldw_bc2 -= dldw_bc0;
    //
    // backward through ray-triangle intersection: d_t=0, d_u=dldw_bc1, d_v=dldw_bc2
    const auto res = tri3::intersection_against_line_bwd_wrt_tri(
        p0, p1, p2, ray.first.data(), ray.second.data(), 0.f, dldw_bc1, dldw_bc2);
    if (!res) { return; }
    const float bc1 = res->u;
    const float bc2 = res->v;
    const float bc0 = 1.f - bc1 - bc2;
    //
    // accumulate dldw_vtx2xyz (atomicAdd: multiple pixels may share a vertex)
    for (uint32_t i = 0; i < 3; ++i) {
        atomicAdd(dldw_vtx2xyz + i0 * 3 + i, res->d_p0[i]);
        atomicAdd(dldw_vtx2xyz + i1 * 3 + i, res->d_p1[i]);
        atomicAdd(dldw_vtx2xyz + i2 * 3 + i, res->d_p2[i]);
    }
    //
    // accumulate dldw_vtx2val (atomicAdd: same reason)
    for (uint32_t i = 0; i < num_vdim; ++i) {
        const float dl = dldw_pix2val[i_pix * num_vdim + i];
        atomicAdd(dldw_vtx2val + i0 * num_vdim + i, dl * bc0);
        atomicAdd(dldw_vtx2val + i1 * num_vdim + i, dl * bc1);
        atomicAdd(dldw_vtx2val + i2 * num_vdim + i, dl * bc2);
    }
}

}