#include <cstdint>
#include <cuda_runtime.h>
#include "del_geo/aabb3.h"
#include "del_geo/tet.h"
#include "del_geo/prism.h"
#include "del_geo/hex.h"
#include "del_geo/pyramid.h"



// -------------------------------------------------------
// Dispatch: is query point inside this element?
// -------------------------------------------------------
__device__
static bool parametric_coord(
    float pco[3],
    const uint32_t* node2vtx, const uint32_t num_node,
    const float* vtx2xyz, const float* query) 
{
    float pts[8][3];
    for (uint32_t n=0;n<num_node;++n) {
        uint32_t iv=node2vtx[n];
        for (int d=0;d<3;++d) pts[n][d]=vtx2xyz[iv*3+d]-query[d];
    }
    switch (num_node) {
        case 4: {
          return tet::tet_parametric(pco,pts[0],pts[1],pts[2],pts[3]);
        }
        case 5: {
          return pyramid::parametric_coord_for_origin(pco,pts[0],pts[1],pts[2],pts[3],pts[4]);
        }
        case 6: {
          return prism::parametric_coord_for_origin(pco,pts[0],pts[1],pts[2],pts[3],pts[4],pts[5]);
        }
        case 8: {
          return hex::parametric_coord_for_origin(pco,pts[0],pts[1],pts[2],pts[3],
                                          pts[4],pts[5],pts[6],pts[7]);
        }
        default: return false;
    }
}

// -------------------------------------------------------
// Main kernel: one thread per query point
// -------------------------------------------------------
extern "C" {

__global__
void search_elem_contain_points(
    const uint32_t  num_elem,
    const uint32_t* elem2idx_offset,
    const uint32_t* idx2vtx,
    const float*    vtx2xyz,
    const uint32_t* bvhnodes,
    const float*    bvhnode2aabb,
    const uint32_t  num_wtx,
    const float*    wtx2xyz,
    uint32_t*       wtx2elem,
    float*          wtx2param)
{
    const uint32_t i_wtx = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_wtx >= num_wtx) { return; }


    const float* q = wtx2xyz + i_wtx * 3;
    wtx2elem[i_wtx] = UINT32_MAX;

/*
    for(uint32_t i_elem=0;i_elem<num_elem;++i_elem){
        const uint32_t i0 = elem2idx_offset[i_elem];
        const uint32_t i1 = elem2idx_offset[i_elem+1];
        float pco[3];
        if (parametric_coord(pco, idx2vtx+i0, i1-i0, vtx2xyz, q)) {
            wtx2elem[i_wtx]      = i_elem;
            wtx2param[i_wtx*3+0] = pco[0];
            wtx2param[i_wtx*3+1] = pco[1];
            wtx2param[i_wtx*3+2] = pco[2];
            return;
        }
    }
*/

    wtx2param[i_wtx*3+0] = 0.f;
    wtx2param[i_wtx*3+1] = 0.f;
    wtx2param[i_wtx*3+2] = 0.f;

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;  // root node

    while (top > 0) {
        const uint32_t i_node = stack[--top];
        const float* aabb = bvhnode2aabb + i_node * 6;
        if( !aabb3::is_include_point(aabb, q) ) continue;

        if (bvhnodes[i_node*3+2] == UINT32_MAX) {
            // leaf — test actual element
            const uint32_t i_elem = bvhnodes[i_node*3+1];
            const uint32_t i0 = elem2idx_offset[i_elem];
            const uint32_t i1 = elem2idx_offset[i_elem+1];
            float pco[3];
            if (parametric_coord(pco, idx2vtx+i0, i1-i0, vtx2xyz, q)) {
                wtx2elem[i_wtx]      = i_elem;
                wtx2param[i_wtx*3+0] = pco[0];
                wtx2param[i_wtx*3+1] = pco[1];
                wtx2param[i_wtx*3+2] = pco[2];
                return;
            }
        } else {
            // branch — push children (right then left so left is popped first)
            stack[top++] = bvhnodes[i_node*3+2];
            stack[top++] = bvhnodes[i_node*3+1];
        }
    }

}

__global__
void interpolate_values_at_points(
  const uint32_t num_wtx,
  const uint32_t* elem2idx_offset,
  const uint32_t* idx2vtx,
  const float* vtx2val,
  const uint32_t* wtx2elem,
  const float* wtx2param,
  const uint32_t num_value_dim,
  float* wtx2val_out)
{
    const uint32_t i_wtx = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_wtx >= num_wtx) return;

    float* out = wtx2val_out + i_wtx * num_value_dim;
    for (uint32_t d = 0; d < num_value_dim; ++d) out[d] = 0.f;

    const uint32_t i_elem = wtx2elem[i_wtx];
    if (i_elem == UINT32_MAX) return;

    const uint32_t i0 = elem2idx_offset[i_elem];
    const uint32_t i1 = elem2idx_offset[i_elem + 1];
    const uint32_t num_node = i1 - i0;
    const float* pco = wtx2param + i_wtx * 3;
    const float r = pco[0], s = pco[1], t = pco[2];

    float sf[8];
    switch (num_node) {
        case 4:
            sf[0] = r; sf[1] = s; sf[2] = t; sf[3] = 1.f - r - s - t;
            break;
        case 5: { // pyramid: N0=(1-r)(1-s)(1-t), N1=r(1-s)(1-t), N2=rs(1-t), N3=(1-r)s(1-t), N4=t
            float rm=1-r, sm=1-s, tm=1-t;
            sf[0]=rm*sm*tm; sf[1]=r*sm*tm; sf[2]=r*s*tm; sf[3]=rm*s*tm; sf[4]=t;
            break;
        }
        case 6: { // prism: N0=(1-r-s)(1-t), N1=r(1-t), N2=s(1-t), N3=(1-r-s)t, N4=rt, N5=st
            float rs=1-r-s, tm=1-t;
            sf[0]=rs*tm; sf[1]=r*tm; sf[2]=s*tm; sf[3]=rs*t; sf[4]=r*t; sf[5]=s*t;
            break;
        }
        case 8: { // hex: Ni = (1/8)(1±r)(1±s)(1±t)
            float e=0.125f;
            sf[0]=e*(1-r)*(1-s)*(1-t); sf[1]=e*(1+r)*(1-s)*(1-t);
            sf[2]=e*(1+r)*(1+s)*(1-t); sf[3]=e*(1-r)*(1+s)*(1-t);
            sf[4]=e*(1-r)*(1-s)*(1+t); sf[5]=e*(1+r)*(1-s)*(1+t);
            sf[6]=e*(1+r)*(1+s)*(1+t); sf[7]=e*(1-r)*(1+s)*(1+t);
            break;
        }
        default: return;
    }

    for (uint32_t i_node = 0; i_node < num_node; ++i_node) {
        const uint32_t i_vtx = idx2vtx[i0 + i_node];
        const float* val = vtx2val + i_vtx * num_value_dim;
        for (uint32_t d = 0; d < num_value_dim; ++d)
            out[d] += sf[i_node] * val[d];
    }
}


} // extern "C"
