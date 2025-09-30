#include <cstdint> // for uint32_t
#include <stdio.h>
#include <cuda_runtime.h>
//
#include "vec3.h"
#include "tri3.h"

extern "C" {

__global__
void tri2normal(
    const uint32_t num_tri,
    const int32_t *tri2vtx,
    const float *vtx2xyz,
    float *tri2nrm)
{
    int i_tri = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_tri >= num_tri) { return; }
    //
    const float* p0 = vtx2xyz + tri2vtx[i_tri*3+0]*3;
    const float* p1 = vtx2xyz + tri2vtx[i_tri*3+1]*3;
    const float* p2 = vtx2xyz + tri2vtx[i_tri*3+2]*3;
    const auto n = tri3::normal(p0, p1, p2);
    tri2nrm[i_tri*3+0] = n[0];
    tri2nrm[i_tri*3+1] = n[1];
    tri2nrm[i_tri*3+2] = n[2];
    // printf("%d\n", i_tri);
}

__global__
void bwd_tri2normal(
    const uint32_t num_tri,
    const int32_t *tri2vtx,
    const float *vtx2xyz,
    const float *dw_tri2nrm,
    float *dw_vtx2xyz)
{
    int i_tri = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_tri >= num_tri) { return; }
    //
    const int32_t i0_vtx = tri2vtx[i_tri*3+0];
    const int32_t i1_vtx = tri2vtx[i_tri*3+1];
    const int32_t i2_vtx = tri2vtx[i_tri*3+2];
    const float* p0 = vtx2xyz + i0_vtx*3;
    const float* p1 = vtx2xyz + i1_vtx*3;
    const float* p2 = vtx2xyz + i2_vtx*3;
    const auto dw = tri3::dw_normal(p0, p1, p2);
    const float* dw_nrm = dw_tri2nrm + i_tri * 3;
    const auto q0 = vec3::mult_mat3_col_major(dw_nrm, dw.d_p0.data());
    const auto q1 = vec3::mult_mat3_col_major(dw_nrm, dw.d_p1.data());
    const auto q2 = vec3::mult_mat3_col_major(dw_nrm, dw.d_p2.data());
    atomicAdd(dw_vtx2xyz + i0_vtx * 3+0, q0[0]);
    atomicAdd(dw_vtx2xyz + i0_vtx * 3+1, q0[1]);
    atomicAdd(dw_vtx2xyz + i0_vtx * 3+2, q0[2]);
    atomicAdd(dw_vtx2xyz + i1_vtx * 3+0, q1[0]);
    atomicAdd(dw_vtx2xyz + i1_vtx * 3+1, q1[1]);
    atomicAdd(dw_vtx2xyz + i1_vtx * 3+2, q1[2]);
    atomicAdd(dw_vtx2xyz + i2_vtx * 3+0, q2[0]);
    atomicAdd(dw_vtx2xyz + i2_vtx * 3+1, q2[1]);
    atomicAdd(dw_vtx2xyz + i2_vtx * 3+2, q2[2]);
}

}