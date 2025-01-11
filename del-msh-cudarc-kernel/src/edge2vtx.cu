
#include "mat4_col_major.h"
#include "tri3.h"

extern "C" {
__global__
void edge2vtx_contour_set_flag(
  const uint32_t num_edge,
  uint32_t *edge2flag,
  const uint32_t *edge2vtx,
  const uint32_t *edge2tri,
  const uint32_t *tri2vtx,
  const float *vtx2xyz,
  const float *transform_world2ndc,
  const float *transform_ndc2world)
{
    int i_edge = blockDim.x * blockIdx.x + threadIdx.x;
    if( i_edge >= num_edge ){ return; }
    //
    const uint32_t i0_vtx = edge2vtx[i_edge*2+0];
    const uint32_t i1_vtx = edge2vtx[i_edge*2+1];
    const float pos_mid[3] = {
        (vtx2xyz[i0_vtx*3+0] + vtx2xyz[i1_vtx*3+0])*0.5f,
        (vtx2xyz[i0_vtx*3+1] + vtx2xyz[i1_vtx*3+1])*0.5f,
        (vtx2xyz[i0_vtx*3+2] + vtx2xyz[i1_vtx*3+2])*0.5f
    };
    const auto ray = mat4_col_major::ray_from_transform_world2ndc(
        transform_world2ndc,
        pos_mid,
        transform_ndc2world
    );
    // -----
    bool flg0;
    {
        const uint32_t i_tri = edge2tri[i_edge * 2];
        const float* p0 = vtx2xyz + tri2vtx[i_tri*3+0]*3;
        const float* p1 = vtx2xyz + tri2vtx[i_tri*3+1]*3;
        const float* p2 = vtx2xyz + tri2vtx[i_tri*3+2]*3;
        const auto n0 = tri3::normal(p0, p1, p2);
        flg0 = vec3::dot(n0.data(),ray.second.data()) > 0.f;
    }
    bool flg1;
    {
        const uint32_t i_tri = edge2tri[i_edge * 2 + 1];
        const float* p0 = vtx2xyz + tri2vtx[i_tri*3+0]*3;
        const float* p1 = vtx2xyz + tri2vtx[i_tri*3+1]*3;
        const float* p2 = vtx2xyz + tri2vtx[i_tri*3+2]*3;
        const auto n0 = tri3::normal(p0, p1, p2);
        flg1 = vec3::dot(n0.data(),ray.second.data()) > 0.f;
    }
    if(flg0 == flg1 ){ return; }
    edge2flag[i_edge] = 1;
}

}