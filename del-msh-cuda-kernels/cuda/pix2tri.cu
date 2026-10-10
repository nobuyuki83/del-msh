#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <cfloat>
#include "del_geo/mat4_col_major.h"
#include "del_geo/tri3.h"
#include "del_geo/aabb.h"
#include "ray_for_pixel.h"
#include "tri2_scanline.h"

// Like mat4_col_major::transform_homogeneous but also returns the raw clip-w.
// Returns false when hw == 0 (degenerate).
__device__
bool transform_homogeneous_hw(
    const float *transform, const float *x,
    float ndc[3], float &hw)
{
    float hx = transform[0]*x[0] + transform[4]*x[1] + transform[8]*x[2]  + transform[12];
    float hy = transform[1]*x[0] + transform[5]*x[1] + transform[9]*x[2]  + transform[13];
    float hz = transform[2]*x[0] + transform[6]*x[1] + transform[10]*x[2] + transform[14];
    hw       = transform[3]*x[0] + transform[7]*x[1] + transform[11]*x[2] + transform[15];
    if (hw == 0.f) return false;
    float inv = 1.f / hw;
    ndc[0] = hx * inv;  ndc[1] = hy * inv;  ndc[2] = hz * inv;
    return true;
}

// Linearly interpolate two homogeneous vertices at parameter t.
// Interpolation is done in clip space (multiply by hw, lerp, divide back).
__device__
void interp_hom(
    const float ndc_a[3], float hw_a,
    const float ndc_b[3], float hw_b,
    float t,
    float ndc_c[3], float &hw_c)
{
    hw_c = hw_a + t * (hw_b - hw_a);
    float inv = 1.f / hw_c;
    for (int i = 0; i < 3; ++i)
        ndc_c[i] = (ndc_a[i]*hw_a + t*(ndc_b[i]*hw_b - ndc_a[i]*hw_a)) * inv;
}

// Convert a float to an unsigned integer that preserves the float's total order.
// Positive floats map to [0x80000000, 0xFFFFFFFF]; negatives map to [0, 0x7FFFFFFF].
// -inf (0xFF800000) maps to 0x007FFFFF — the lowest possible value.
__device__ inline uint32_t float_to_orderable(float f) {
    uint32_t bits = __float_as_uint(f);
    return (bits >> 31) ? ~bits : (bits ^ 0x80000000u);
}

// Rasterize one projected triangle into a packed depth+tri buffer.
// Each pixel slot holds (orderable_depth << 32) | tri_id.
// atomicMax ensures the frontmost (highest NDC-z) triangle wins with no races.
__device__
void rasterize_projected_tri(
    unsigned long long *pix2packed,
    const uint32_t img_w,
    const uint32_t img_h,
    const uint32_t i_tri_val,
    const float ndc[3][3],   // ndc[vertex][x/y/z]
    const float hw[3])       // clip-space w for each vertex
{
    // NDC → pixel coordinates  (origin top-left, y down)
    float px[3][2];
    for (int i = 0; i < 3; ++i) {
        px[i][0] = (ndc[i][0] + 1.f) * 0.5f * (float)img_w;
        px[i][1] = (1.f - ndc[i][1]) * 0.5f * (float)img_h;
        if (!isfinite(px[i][0]) || !isfinite(px[i][1])) { return; }
    }

    // Bounding-box cull
    float min_x = fminf(fminf(px[0][0], px[1][0]), px[2][0]);
    float max_x = fmaxf(fmaxf(px[0][0], px[1][0]), px[2][0]);
    float min_y = fminf(fminf(px[0][1], px[1][1]), px[2][1]);
    float max_y = fmaxf(fmaxf(px[0][1], px[1][1]), px[2][1]);
    if (max_x < 0.f || min_x >= (float)img_w ||
        max_y < 0.f || min_y >= (float)img_h) { return; }

    // Precompute signed 2× area for barycentric coordinates.
    float area2 = (px[1][0]-px[0][0])*(px[2][1]-px[0][1])
                - (px[1][1]-px[0][1])*(px[2][0]-px[0][0]);
    if (fabsf(area2) < 1e-10f) { return; }
    float inv_area2 = 1.f / area2;

    tri2_scanline::TriangleScanlineIter it(px[0], px[1], px[2]);
    int32_t ix, iy;
    while (it.next(ix, iy)) {
        if (ix < 0 || iy < 0 || ix >= (int32_t)img_w || iy >= (int32_t)img_h) { continue; }

        float qx = (float)ix + 0.5f;
        float qy = (float)iy + 0.5f;

        // Barycentric coords via signed sub-areas
        float b0 = ((px[1][0]-qx)*(px[2][1]-qy) - (px[1][1]-qy)*(px[2][0]-qx)) * inv_area2;
        float b1 = ((px[2][0]-qx)*(px[0][1]-qy) - (px[2][1]-qy)*(px[0][0]-qx)) * inv_area2;
        float b2 = 1.f - b0 - b1;
        if (b0 < 0.f || b1 < 0.f || b2 < 0.f) { continue; }

        // Perspective-correct depth
        float q0 = b0 / hw[0];
        float q1 = b1 / hw[1];
        float q2 = b2 / hw[2];
        float inv_sum = 1.f / (q0 + q1 + q2);
        float depth = (q0*ndc[0][2] + q1*ndc[1][2] + q2*ndc[2][2]) * inv_sum;

        // Pack orderable depth (upper 32 bits) with tri_id (lower 32 bits) and
        // atomically keep the maximum — the frontmost triangle wins.
        unsigned long long new_val =
            ((unsigned long long)float_to_orderable(depth) << 32) | i_tri_val;
        atomicMax(&pix2packed[(uint32_t)iy * img_w + (uint32_t)ix], new_val);
    }
}

extern "C"{

__global__
void pix2tri_from_raycast(
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
                if( depth >= 0.f && depth < hit_depth ){
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

// Fill pix2packed with the background sentinel:
//   upper 32 bits = orderable(-inf) = 0x007FFFFF  (smallest orderable value)
//   lower 32 bits = UINT32_MAX                     (background tri_id)
// Combined: 0x007FFFFFFFFFFFFFull
__global__
void pix2packed_init(unsigned long long *pix2packed, const uint32_t num_pix) {
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_pix) return;
    pix2packed[i] = 0x007FFFFFFFFFFFFFull;
}

// Unpack pix2packed back into the caller-owned pix2tri / pix2depth arrays.
__global__
void pix2tri_unpack(
    uint32_t             *pix2tri,
    float                *pix2depth,
    const unsigned long long *pix2packed,
    const uint32_t        num_pix)
{
    const uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_pix) return;
    const unsigned long long packed = pix2packed[i];
    const uint32_t tri_id   = (uint32_t)(packed & 0xFFFFFFFFull);
    const uint32_t orderable = (uint32_t)(packed >> 32);
    // Invert float_to_orderable: if bit31 was set the float was positive.
    const uint32_t float_bits = (orderable >> 31) ? (orderable ^ 0x80000000u) : ~orderable;
    pix2tri[i]   = tri_id;
    pix2depth[i] = __uint_as_float(float_bits);
}

// One thread per triangle.  Rasterizes into a packed depth+tri buffer using
// atomicMax so concurrent writes to the same pixel are race-free.
__global__
void pix2tri_from_rasterization(
    unsigned long long *pix2packed,
    const uint32_t      num_tri,
    const uint32_t     *tri2vtx,
    const float        *vtx2xyz,
    const uint32_t      img_w,
    const uint32_t      img_h,
    const float        *transform_world2ndc)
{
    const uint32_t i_tri = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_tri >= num_tri) return;

    const uint32_t i0 = tri2vtx[i_tri*3 + 0];
    const uint32_t i1 = tri2vtx[i_tri*3 + 1];
    const uint32_t i2 = tri2vtx[i_tri*3 + 2];

    float ndc[3][3], hw[3];
    if (!transform_homogeneous_hw(transform_world2ndc, vtx2xyz + i0*3, ndc[0], hw[0])) return;
    if (!transform_homogeneous_hw(transform_world2ndc, vtx2xyz + i1*3, ndc[1], hw[1])) return;
    if (!transform_homogeneous_hw(transform_world2ndc, vtx2xyz + i2*3, ndc[2], hw[2])) return;

    // m[11]: z→w element; +1 (Blender) or -1 (OpenGL) for perspective, 0 for ortho.
    // hw * t11 < 0 means "in front of the camera" for either convention.
    const float t11 = transform_world2ndc[11];

    if (t11 != 0.f) {
        // Skip triangles with any vertex behind the camera plane.
        if (hw[0]*t11 >= 0.f || hw[1]*t11 >= 0.f || hw[2]*t11 >= 0.f) return;
    }

    // Near-clip score: c >= 0 means vertex is at or beyond the near clip plane.
    const float c0 = hw[0] * (ndc[0][2] - t11);
    const float c1 = hw[1] * (ndc[1][2] - t11);
    const float c2 = hw[2] * (ndc[2][2] - t11);

    auto clip_t = [](float ca, float cb) { return ca / (ca - cb); };

    if (t11 == 0.f || (c0 >= 0.f && c1 >= 0.f && c2 >= 0.f)) {
        // Orthographic or all vertices beyond the near clip: rasterize whole triangle.
        rasterize_projected_tri(pix2packed, img_w, img_h, i_tri,
                                ndc, hw);
    } else if (c0 < 0.f && c1 < 0.f && c2 < 0.f) {
        // All between camera and near clip: skip.
    } else {
        // Mixed: clip against the near plane.
        const float cs[3] = { c0, c1, c2 };

        int n_valid = (c0 >= 0.f ? 1 : 0) + (c1 >= 0.f ? 1 : 0) + (c2 >= 0.f ? 1 : 0);

        if (n_valid == 1) {
            // 1 inside vertex → 1 output triangle.
            int ia = (c0 >= 0.f) ? 0 : (c1 >= 0.f) ? 1 : 2;
            int ib = (ia + 1) % 3;
            int ic = (ia + 2) % 3;

            float ndc_ab[3], ndc_ac[3]; float hw_ab, hw_ac;
            interp_hom(ndc[ia], hw[ia], ndc[ib], hw[ib], clip_t(cs[ia], cs[ib]), ndc_ab, hw_ab);
            interp_hom(ndc[ia], hw[ia], ndc[ic], hw[ic], clip_t(cs[ia], cs[ic]), ndc_ac, hw_ac);

            const float ndc_out[3][3] = {{ndc[ia][0],ndc[ia][1],ndc[ia][2]},
                                         {ndc_ab[0],ndc_ab[1],ndc_ab[2]},
                                         {ndc_ac[0],ndc_ac[1],ndc_ac[2]}};
            const float hw_out[3] = { hw[ia], hw_ab, hw_ac };
            rasterize_projected_tri(pix2packed, img_w, img_h, i_tri,
                                    ndc_out, hw_out);
        } else {
            // 2 inside vertices → 2 output triangles.
            int ic = (c0 < 0.f) ? 0 : (c1 < 0.f) ? 1 : 2;
            int ia = (ic + 1) % 3;
            int ib = (ic + 2) % 3;

            float ndc_ac[3], ndc_bc[3]; float hw_ac, hw_bc;
            interp_hom(ndc[ia], hw[ia], ndc[ic], hw[ic], clip_t(cs[ia], cs[ic]), ndc_ac, hw_ac);
            interp_hom(ndc[ib], hw[ib], ndc[ic], hw[ic], clip_t(cs[ib], cs[ic]), ndc_bc, hw_bc);

            const float ndc_t1[3][3] = {{ndc[ia][0],ndc[ia][1],ndc[ia][2]},
                                         {ndc[ib][0],ndc[ib][1],ndc[ib][2]},
                                         {ndc_ac[0],ndc_ac[1],ndc_ac[2]}};
            const float hw_t1[3] = { hw[ia], hw[ib], hw_ac };
            rasterize_projected_tri(pix2packed, img_w, img_h, i_tri,
                                    ndc_t1, hw_t1);

            const float ndc_t2[3][3] = {{ndc[ib][0],ndc[ib][1],ndc[ib][2]},
                                         {ndc_bc[0],ndc_bc[1],ndc_bc[2]},
                                         {ndc_ac[0],ndc_ac[1],ndc_ac[2]}};
            const float hw_t2[3] = { hw[ib], hw_bc, hw_ac };
            rasterize_projected_tri(pix2packed, img_w, img_h, i_tri,
                                    ndc_t2, hw_t2);
        }
    }
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