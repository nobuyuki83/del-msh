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
static bool parametric_coord(float pco[3],
                              const uint32_t* node2vtx, const uint32_t num_node,
                              const float* vtx2xyz, const float* query) {
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

} // extern "C"
