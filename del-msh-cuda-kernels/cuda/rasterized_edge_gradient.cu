#include <cuda_runtime.h>
#include <cfloat>
#include "del_geo/mat4_col_major.h"
#include "del_geo/tri2.h"

extern "C" {

// Returns true if the 2D pixel center (px, py) is inside the projected triangle
// identified by itri, or if itri == UINT32_MAX (background), or if the
// projected triangle is degenerate. Mirrors Rust fn_barycentric + fn_inside.
__device__ bool is_pix_inside_tri(
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    const float *transform_world2pix,
    float px, float py,
    uint32_t itri)
{
    if (itri == UINT32_MAX) { return true; }
    const uint32_t i0 = tri2vtx[itri * 3 + 0];
    const uint32_t i1 = tri2vtx[itri * 3 + 1];
    const uint32_t i2 = tri2vtx[itri * 3 + 2];
    // project vertices from 3D world to 2D pixel space
    const auto r0 = mat4_col_major::transform_homogeneous(transform_world2pix, vtx2xyz + i0 * 3);
    const auto r1 = mat4_col_major::transform_homogeneous(transform_world2pix, vtx2xyz + i1 * 3);
    const auto r2 = mat4_col_major::transform_homogeneous(transform_world2pix, vtx2xyz + i2 * 3);
    const float p0[2] = {r0[0], r0[1]};
    const float p1[2] = {r1[0], r1[1]};
    const float p2[2] = {r2[0], r2[1]};
    const float area = tri2::area(p0, p1, p2);
    if (area == 0.f) { return true; }  // degenerate triangle → treat as inside
    const float inv_area = 1.f / area;
    const float q[2] = {px, py};
    const float b0 = tri2::area(q,  p1, p2) * inv_area;
    const float b1 = tri2::area(p0, q,  p2) * inv_area;
    const float b2 = 1.f - b0 - b1;
    return (b0 >= 0.f && b1 >= 0.f && b2 >= 0.f)
        || (b0 <= 0.f && b1 <= 0.f && b2 <= 0.f);
}

// Horizontal edges: edge i_hedge = ih0 * img_w + iw connects pixel (iw, ih0)
// and pixel (iw, ih0+1).  Output arrays have size (img_h-1) * img_w.
// Thread (x=iw, y=ih0) covers one horizontal edge.
__global__
void hedge_gradient_and_type(
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    const float *transform_world2pix,
    const uint32_t img_w,
    const uint32_t img_h,
    const uint32_t *pix2tri,
    const uint32_t num_vdim,
    const float *pix2val,
    const float *dldw_pix2val,
    uint8_t *hedge2type,
    float *hedge2dldr)
{
    const uint32_t iw  = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ih0 = blockDim.y * blockIdx.y + threadIdx.y;
    if (iw >= img_w || ih0 >= img_h - 1) { return; }

    const uint32_t ih1   = ih0 + 1;
    const uint32_t ipix0 = ih0 * img_w + iw;  // north pixel
    const uint32_t ipix1 = ih1 * img_w + iw;  // south pixel
    const uint32_t i_hedge = ih0 * img_w + iw;
    const uint32_t itri0 = pix2tri[ipix0];
    const uint32_t itri1 = pix2tri[ipix1];
    if (itri0 == itri1) {
        hedge2type[i_hedge] = 0;
    } else {
        const float px0 = iw + 0.5f, py0 = (float)ih0 + 0.5f;
        const float px1 = iw + 0.5f, py1 = (float)ih1 + 0.5f;
        const bool in0_tri1 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px0, py0, itri1);
        const bool in1_tri0 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px1, py1, itri0);
        if      (!in0_tri1 && !in1_tri0) hedge2type[i_hedge] = 1;
        else if ( in0_tri1 && !in1_tri0) hedge2type[i_hedge] = 2;
        else if (!in0_tri1 &&  in1_tri0) hedge2type[i_hedge] = 3;
        else                             hedge2type[i_hedge] = 4;
    }
    float dldr = 0.f;
    for (uint32_t i = 0; i < num_vdim; ++i) {
        const float val0  = pix2val[ipix0 * num_vdim + i];
        const float val1  = pix2val[ipix1 * num_vdim + i];
        const float dval0 = dldw_pix2val[ipix0 * num_vdim + i];
        const float dval1 = dldw_pix2val[ipix1 * num_vdim + i];
        dldr += (dval0 + dval1) * 0.5f * (val0 - val1);
    }
    hedge2dldr[i_hedge] = dldr;
}

// Vertical edges: edge i_vedge = ih * (img_w-1) + iw0 connects pixel (iw0, ih)
// and pixel (iw0+1, ih).  Output arrays have size img_h * (img_w-1).
// Thread (x=iw0, y=ih) covers one vertical edge.
__global__
void vedge_gradient_and_type(
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    const float *transform_world2pix,
    const uint32_t img_w,
    const uint32_t img_h,
    const uint32_t *pix2tri,
    const uint32_t num_vdim,
    const float *pix2val,
    const float *dldw_pix2val,
    uint8_t *vedge2type,
    float *vedge2dldr)
{
    const uint32_t iw0 = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ih  = blockDim.y * blockIdx.y + threadIdx.y;
    if (iw0 >= img_w - 1 || ih >= img_h) { return; }
    const uint32_t iw1    = iw0 + 1;
    const uint32_t ipix0  = ih * img_w + iw0;  // west pixel
    const uint32_t ipix1  = ih * img_w + iw1;  // east pixel
    const uint32_t i_vedge = ih * (img_w - 1) + iw0;
    const uint32_t itri0 = pix2tri[ipix0];
    const uint32_t itri1 = pix2tri[ipix1];
    if (itri0 == itri1) {
        vedge2type[i_vedge] = 0;
    } else {
        const float px0 = (float)iw0 + 0.5f, py0 = ih + 0.5f;
        const float px1 = (float)iw1 + 0.5f, py1 = ih + 0.5f;
        const bool in0_tri1 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px0, py0, itri1);
        const bool in1_tri0 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px1, py1, itri0);
        if      (!in0_tri1 && !in1_tri0) vedge2type[i_vedge] = 1;
        else if ( in0_tri1 && !in1_tri0) vedge2type[i_vedge] = 2;
        else if (!in0_tri1 &&  in1_tri0) vedge2type[i_vedge] = 3;
        else                             vedge2type[i_vedge] = 4;
    }
    float dldr = 0.f;
    for (uint32_t i = 0; i < num_vdim; ++i) {
        const float val0  = pix2val[ipix0 * num_vdim + i];
        const float val1  = pix2val[ipix1 * num_vdim + i];
        const float dval0 = dldw_pix2val[ipix0 * num_vdim + i];
        const float dval1 = dldw_pix2val[ipix1 * num_vdim + i];
        dldr += (dval0 + dval1) * 0.5f * (val0 - val1);
    }
    vedge2dldr[i_vedge] = dldr;
}

__global__
void smooth_hedge_red_black(
    unsigned int img_w,
    unsigned int img_h,
    const unsigned char* __restrict__ hedge2type,
    float* __restrict__ hedge2dldr,
    const unsigned char* __restrict__ vedge2type,
    unsigned int color)
{
    const unsigned int i_hedge = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int num_hedges = (img_h - 1) * img_w;
    if (i_hedge >= num_hedges) { return; }
    const unsigned int iw = i_hedge % img_w;
    const unsigned int ih = i_hedge / img_w;
    if (((iw + ih) & 1u) != color) { return; }  // red-black coloring

    const unsigned char type_c = hedge2type[i_hedge];
    if (type_c == 2 || type_c == 3) { return; }  // fixed-value edges

    float value_sum = 0.0f;
    unsigned int count = 0;

    // North
    if (ih > 0) {
        const unsigned int i_north = (ih - 1) * img_w + iw;
        if (hedge2type[i_north] != 2) {
            value_sum += hedge2dldr[i_north];
            count += 1;
        }
    }

    // South
    if (ih + 1 < img_h - 1) {
        const unsigned int i_south = (ih + 1) * img_w + iw;
        if (hedge2type[i_south] != 3) {
            value_sum += hedge2dldr[i_south];
            count += 1;
        }
    }

    // West
    if (iw > 0) {
        const unsigned int i_vedge_nw = ih * (img_w - 1) + iw - 1;
        const unsigned int i_vedge_sw = (ih + 1) * (img_w - 1) + iw - 1;
        const unsigned char type_nw = vedge2type[i_vedge_nw];
        const unsigned char type_sw = vedge2type[i_vedge_sw];
        if (type_nw != 2 && type_nw != 3 && type_sw != 2 && type_sw != 3) {
            value_sum += hedge2dldr[ih * img_w + iw - 1];
            count += 1;
        }
    }

    // East
    if (iw + 1 < img_w) {
        const unsigned int i_vedge_ne = ih * (img_w - 1) + iw;
        const unsigned int i_vedge_se = (ih + 1) * (img_w - 1) + iw;
        const unsigned char type_ne = vedge2type[i_vedge_ne];
        const unsigned char type_se = vedge2type[i_vedge_se];
        if (type_ne != 2 && type_ne != 3 && type_se != 2 && type_se != 3) {
            value_sum += hedge2dldr[ih * img_w + iw + 1];
            count += 1;
        }
    }

    if (count > 0) {
        hedge2dldr[i_hedge] = value_sum / static_cast<float>(count);
    }
}


__global__
void smooth_vedge_red_black(
    unsigned int img_w,
    unsigned int img_h,
    const unsigned char* __restrict__ vedge2type,
    float* __restrict__ vedge2dldr,
    const unsigned char* __restrict__ hedge2type,
    unsigned int color)
{
    const unsigned int vedge_w = img_w - 1;
    const unsigned int i_vedge = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_vedge >= img_h * vedge_w) { return; }
    const unsigned int iw = i_vedge % vedge_w;
    const unsigned int ih = i_vedge / vedge_w;
    if (((iw + ih) & 1u) != color) { return; }  // red-black coloring

    const unsigned char type_c = vedge2type[i_vedge];
    if (type_c == 2 || type_c == 3) { return; }  // fixed-value edges

    float value_sum = 0.0f;
    unsigned int count = 0;

    // West
    if (iw > 0) {
        if (vedge2type[i_vedge - 1] != 2) {
            value_sum += vedge2dldr[i_vedge - 1];
            count += 1;
        }
    }

    // East
    if (iw + 1 < vedge_w) {
        if (vedge2type[i_vedge + 1] != 3) {
            value_sum += vedge2dldr[i_vedge + 1];
            count += 1;
        }
    }

    // North
    if (ih > 0) {
        const unsigned int i_hedge_nw = (ih - 1) * img_w + iw;
        const unsigned char type_nw = hedge2type[i_hedge_nw];
        const unsigned char type_ne = hedge2type[i_hedge_nw + 1];
        if (type_nw != 2 && type_nw != 3 && type_ne != 2 && type_ne != 3) {
            value_sum += vedge2dldr[i_vedge - vedge_w];
            count += 1;
        }
    }

    // South
    if (ih + 1 < img_h) {
        const unsigned int i_hedge_sw = ih * img_w + iw;
        const unsigned char type_sw = hedge2type[i_hedge_sw];
        const unsigned char type_se = hedge2type[i_hedge_sw + 1];
        if (type_sw != 2 && type_sw != 3 && type_se != 2 && type_se != 3) {
            value_sum += vedge2dldr[i_vedge + vedge_w];
            count += 1;
        }
    }

    if (count > 0) {
        vedge2dldr[i_vedge] = value_sum / static_cast<float>(count);
    }
}

// One thread per vertex.  Bilinearly interpolates x-velocity from vertical
// edges (vedge2vx, sampled at (iw+1, ih+0.5)) and y-velocity from horizontal
// edges (hedge2vy, sampled at (iw+0.5, ih+1)).
__global__
void interpolate_staggered_grid(
    unsigned int img_w,
    unsigned int img_h,
    const float* __restrict__ hedge2vy,
    const float* __restrict__ vedge2vx,
    const float* __restrict__ vtx2xy,
    float* __restrict__ vtx2velo,
    unsigned int num_vtx)
{
    const unsigned int i_vtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_vtx >= num_vtx) { return; }
    const float px = vtx2xy[i_vtx * 2];
    const float py = vtx2xy[i_vtx * 2 + 1];

    // x-velocity: bilinear from vertical edges at (iw0+1.0, ih+0.5)
    {
        const float gx = px - 1.0f;
        const float gy = py - 0.5f;
        const int ix0 = max(0, min((int)floorf(gx), (int)img_w - 2));
        const int iy0 = max(0, min((int)floorf(gy), (int)img_h - 2));
        const int ix1 = min(ix0 + 1, (int)img_w - 2);
        const int iy1 = min(iy0 + 1, (int)img_h - 1);
        const float tx = fminf(fmaxf(gx - (float)ix0, 0.f), 1.f);
        const float ty = fminf(fmaxf(gy - (float)iy0, 0.f), 1.f);
        const int w = (int)img_w - 1;
        vtx2velo[i_vtx * 2] =
            (1.f - tx) * (1.f - ty) * vedge2vx[iy0 * w + ix0]
            + tx * (1.f - ty) * vedge2vx[iy0 * w + ix1]
            + (1.f - tx) * ty * vedge2vx[iy1 * w + ix0]
            + tx * ty * vedge2vx[iy1 * w + ix1];
    }

    // y-velocity: bilinear from horizontal edges at (iw+0.5, ih0+1.0)
    {
        const float gx = px - 0.5f;
        const float gy = py - 1.0f;
        const int ix0 = max(0, min((int)floorf(gx), (int)img_w - 2));
        const int iy0 = max(0, min((int)floorf(gy), (int)img_h - 2));
        const int ix1 = min(ix0 + 1, (int)img_w - 1);
        const int iy1 = min(iy0 + 1, (int)img_h - 2);
        const float tx = fminf(fmaxf(gx - (float)ix0, 0.f), 1.f);
        const float ty = fminf(fmaxf(gy - (float)iy0, 0.f), 1.f);
        vtx2velo[i_vtx * 2 + 1] =
            (1.f - tx) * (1.f - ty) * hedge2vy[iy0 * img_w + ix0]
            + tx * (1.f - ty) * hedge2vy[iy0 * img_w + ix1]
            + (1.f - tx) * ty * hedge2vy[iy1 * img_w + ix0]
            + tx * ty * hedge2vy[iy1 * img_w + ix1];
    }
}



// Returns barycentric coordinates of pixel (px, py) in the projected triangle
// `itri`. Returns false if the triangle is degenerate or itri == UINT32_MAX.
__device__ bool barycentric_of_pixel_in_tri(
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    const float *transform_world2pix,
    float px, float py,
    uint32_t itri,
    float bary[3])
{
    if (itri == UINT32_MAX) { return false; }
    const uint32_t i0 = tri2vtx[itri * 3];
    const uint32_t i1 = tri2vtx[itri * 3 + 1];
    const uint32_t i2 = tri2vtx[itri * 3 + 2];
    const auto r0 = mat4_col_major::transform_homogeneous(transform_world2pix, vtx2xyz + i0 * 3);
    const auto r1 = mat4_col_major::transform_homogeneous(transform_world2pix, vtx2xyz + i1 * 3);
    const auto r2 = mat4_col_major::transform_homogeneous(transform_world2pix, vtx2xyz + i2 * 3);
    const float p0[2] = {r0[0], r0[1]};
    const float p1[2] = {r1[0], r1[1]};
    const float p2[2] = {r2[0], r2[1]};
    const float area = tri2::area(p0, p1, p2);
    if (area == 0.f) { return false; }
    const float inv_area = 1.f / area;
    const float q[2] = {px, py};
    bary[0] = tri2::area(q,  p1, p2) * inv_area;
    bary[1] = tri2::area(p0, q,  p2) * inv_area;
    bary[2] = 1.f - bary[0] - bary[1];
    return true;
}

// Jacobian row of one perspective-divided pixel coordinate with respect to a
// world-space vertex. axis=0 selects pixel x and axis=1 selects pixel y.
__device__ bool projection_gradient(
    const float *transform_world2pix,
    const float *xyz,
    const int axis,
    float *dxyz)
{
    const float q = transform_world2pix[axis] * xyz[0]
        + transform_world2pix[axis + 4] * xyz[1]
        + transform_world2pix[axis + 8] * xyz[2]
        + transform_world2pix[axis + 12];
    const float w = transform_world2pix[3] * xyz[0]
        + transform_world2pix[7] * xyz[1]
        + transform_world2pix[11] * xyz[2]
        + transform_world2pix[15];
    if (fabsf(w) <= FLT_EPSILON) { return false; }
    const float inv_w2 = 1.f / (w * w);
    dxyz[0] = (transform_world2pix[axis] * w - q * transform_world2pix[3]) * inv_w2;
    dxyz[1] = (transform_world2pix[axis + 4] * w - q * transform_world2pix[7]) * inv_w2;
    dxyz[2] = (transform_world2pix[axis + 8] * w - q * transform_world2pix[11]) * inv_w2;
    return true;
}

// Backward pass for horizontal edges (each edge connects pixel (iw,ih0) and
// (iw,ih0+1)). Uses the perspective-correct pixel-y projection Jacobian.
// Thread (x=iw, y=ih0), grid covers img_w × (img_h-1).
__global__
void bwd_hedge(
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    float *dldw_vtx2xyz,
    const float *transform_world2pix,
    const uint32_t img_w,
    const uint32_t img_h,
    const uint32_t *pix2tri,
    const uint32_t num_vdim,
    const float *pix2val,
    const float *dldw_pixval)
{
    const uint32_t iw  = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ih0 = blockDim.y * blockIdx.y + threadIdx.y;
    if (iw >= img_w || ih0 >= img_h - 1) { return; }

    const uint32_t ih1   = ih0 + 1;
    const uint32_t ipix0 = ih0 * img_w + iw;
    const uint32_t ipix1 = ih1 * img_w + iw;
    const uint32_t itri0 = pix2tri[ipix0];
    const uint32_t itri1 = pix2tri[ipix1];
    if (itri0 == itri1) { return; }

    const float px0 = iw + 0.5f, py0 = (float)ih0 + 0.5f;
    const float px1 = iw + 0.5f, py1 = (float)ih1 + 0.5f;
    const bool in0_tri1 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px0, py0, itri1);
    const bool in1_tri0 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px1, py1, itri0);
    if (!in0_tri1 && !in1_tri0) { return; }
    if ( in0_tri1 &&  in1_tri0) { return; }  // intersection — todo

    float dldpa = 0.f;
    for (uint32_t i = 0; i < num_vdim; ++i) {
        const float val0  = pix2val[ipix0 * num_vdim + i];
        const float val1  = pix2val[ipix1 * num_vdim + i];
        const float dval0 = dldw_pixval[ipix0 * num_vdim + i];
        const float dval1 = dldw_pixval[ipix1 * num_vdim + i];
        dldpa += (dval0 + dval1) * 0.5f * (val0 - val1);
    }

    uint32_t target_tri;
    float pxq, pyq;
    if (in1_tri0) {
        // south pixel center is inside north tri → south tri receives gradient
        target_tri = itri1;
        pxq = px1; pyq = py1;
    } else {
        // north pixel center is inside south tri → north tri receives gradient
        target_tri = itri0;
        pxq = px0; pyq = py0;
    }

    float bary[3];
    if (!barycentric_of_pixel_in_tri(tri2vtx, vtx2xyz, transform_world2pix, pxq, pyq, target_tri, bary)) {
        return;
    }
    for (int inode = 0; inode < 3; ++inode) {
        const uint32_t ivtx = tri2vtx[target_tri * 3 + inode];
        float dxyz[3];
        if (!projection_gradient(transform_world2pix, vtx2xyz + ivtx * 3, 1, dxyz)) { continue; }
        atomicAdd(&dldw_vtx2xyz[ivtx * 3 + 0], bary[inode] * dxyz[0] * dldpa);
        atomicAdd(&dldw_vtx2xyz[ivtx * 3 + 1], bary[inode] * dxyz[1] * dldpa);
        atomicAdd(&dldw_vtx2xyz[ivtx * 3 + 2], bary[inode] * dxyz[2] * dldpa);
    }
}

// Backward pass for vertical edges (each edge connects pixel (iw0,ih) and
// (iw0+1,ih)). Uses the perspective-correct pixel-x projection Jacobian.
// Thread (x=iw0, y=ih), grid covers (img_w-1) × img_h.
__global__
void bwd_vedge(
    const uint32_t *tri2vtx,
    const float *vtx2xyz,
    float *dldw_vtx2xyz,
    const float *transform_world2pix,
    const uint32_t img_w,
    const uint32_t img_h,
    const uint32_t *pix2tri,
    const uint32_t num_vdim,
    const float *pix2val,
    const float *dldw_pixval)
{
    const uint32_t iw0 = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t ih  = blockDim.y * blockIdx.y + threadIdx.y;
    if (iw0 >= img_w - 1 || ih >= img_h) { return; }

    const uint32_t iw1   = iw0 + 1;
    const uint32_t ipix0 = ih * img_w + iw0;
    const uint32_t ipix1 = ih * img_w + iw1;
    const uint32_t itri0 = pix2tri[ipix0];
    const uint32_t itri1 = pix2tri[ipix1];
    if (itri0 == itri1) { return; }

    const float px0 = (float)iw0 + 0.5f, py0 = ih + 0.5f;
    const float px1 = (float)iw1 + 0.5f, py1 = ih + 0.5f;
    const bool in0_tri1 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px0, py0, itri1);
    const bool in1_tri0 = is_pix_inside_tri(tri2vtx, vtx2xyz, transform_world2pix, px1, py1, itri0);
    if (!in0_tri1 && !in1_tri0) { return; }
    if ( in0_tri1 &&  in1_tri0) { return; }  // intersection — todo

    float dldpa = 0.f;
    for (uint32_t i = 0; i < num_vdim; ++i) {
        const float val0  = pix2val[ipix0 * num_vdim + i];
        const float val1  = pix2val[ipix1 * num_vdim + i];
        const float dval0 = dldw_pixval[ipix0 * num_vdim + i];
        const float dval1 = dldw_pixval[ipix1 * num_vdim + i];
        dldpa += (dval0 + dval1) * 0.5f * (val0 - val1);
    }

    uint32_t target_tri;
    float pxq, pyq;
    if (in1_tri0) {
        // east pixel center is inside west tri → east tri receives gradient
        target_tri = itri1;
        pxq = px1; pyq = py1;
    } else {
        // west pixel center is inside east tri → west tri receives gradient
        target_tri = itri0;
        pxq = px0; pyq = py0;
    }

    float bary[3];
    if (!barycentric_of_pixel_in_tri(tri2vtx, vtx2xyz, transform_world2pix, pxq, pyq, target_tri, bary)) {
        return;
    }
    for (int inode = 0; inode < 3; ++inode) {
        const uint32_t ivtx = tri2vtx[target_tri * 3 + inode];
        float dxyz[3];
        if (!projection_gradient(transform_world2pix, vtx2xyz + ivtx * 3, 0, dxyz)) { continue; }
        atomicAdd(&dldw_vtx2xyz[ivtx * 3 + 0], bary[inode] * dxyz[0] * dldpa);
        atomicAdd(&dldw_vtx2xyz[ivtx * 3 + 1], bary[inode] * dxyz[1] * dldpa);
        atomicAdd(&dldw_vtx2xyz[ivtx * 3 + 2], bary[inode] * dxyz[2] * dldpa);
    }
}

}  // extern "C"
