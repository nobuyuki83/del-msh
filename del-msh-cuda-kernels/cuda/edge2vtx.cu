
#include "del_geo/mat4_col_major.h"
#include "del_geo/tri3.h"

extern "C" {

__global__
void edge2vtx_from_vtx2vtx(
    const uint32_t num_vtx,
    const uint32_t* vtx2idx_offset,
    const uint32_t* idx2vtx,
    uint32_t* edge2vtx
    )
{
    int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i_vtx >= num_vtx ){ return; }
    //
    const auto idx0 = vtx2idx_offset[i_vtx];
    const auto idx1 = vtx2idx_offset[i_vtx + 1];
    for (uint32_t idx = idx0; idx < idx1; ++idx) {
        const auto j_vtx = idx2vtx[idx];
        edge2vtx[idx*2 + 0] = i_vtx;
        edge2vtx[idx*2 + 1] = j_vtx;
    }
}


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


// Compute edge-to-triangle adjacency for a triangle mesh.
// Each thread processes one edge and finds the (at most two) triangles sharing it,
// using the vertex-to-triangle adjacency list (vtx2idx_offset / idx2tri) for lookup.
__global__
void edge2elem_from_edge2vtx_of_tri2vtx(
    const uint32_t num_edge,
    const uint32_t* edge2vtx,
    const uint32_t* tri2vtx,
    const uint32_t* vtx2idx_offset,
    const uint32_t* idx2tri,
    uint32_t* edge2tri)
{
    const int i_edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_edge >= num_edge) { return; }
    //
    const uint32_t i0_vtx = edge2vtx[i_edge * 2 + 0];
    const uint32_t i1_vtx = edge2vtx[i_edge * 2 + 1];
    uint32_t i_cnt = 0;
    // Iterate over triangles adjacent to the first vertex of the edge
    for (uint32_t idx = vtx2idx_offset[i0_vtx]; idx < vtx2idx_offset[i0_vtx + 1]; ++idx) {
        const uint32_t i_tri = idx2tri[idx];
        const uint32_t j0_vtx = tri2vtx[i_tri * 3 + 0];
        const uint32_t j1_vtx = tri2vtx[i_tri * 3 + 1];
        const uint32_t j2_vtx = tri2vtx[i_tri * 3 + 2];
        // Check if this triangle also contains i1_vtx (i.e. shares the edge)
        const bool is_adjacent =
            (i0_vtx == j0_vtx && (j1_vtx == i1_vtx || j2_vtx == i1_vtx)) ||
            (i0_vtx == j1_vtx && (j2_vtx == i1_vtx || j0_vtx == i1_vtx)) ||
            (i0_vtx == j2_vtx && (j0_vtx == i1_vtx || j1_vtx == i1_vtx));
        if (!is_adjacent) { continue; }
        edge2tri[i_edge * 2 + i_cnt] = i_tri;
        ++i_cnt;
        if (i_cnt == 2) { break; }  // At most two triangles share a manifold edge
    }
}

}