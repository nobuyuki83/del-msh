#include <cstdint> // for uint32_t
#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {

__global__
void laplacian_smoothing(
    const uint32_t num_vtx,
    const int32_t *vtx2idx,
    const int32_t *idx2vtx,
    float lambda,
    float *vtx2vars_next,
    const float *vtx2vars_prev,
    const float *vtx2trgs)
{
    int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_vtx >= num_vtx) { return; }
    //
    float rhs[3] = {
        vtx2trgs[i_vtx*3+0],
        vtx2trgs[i_vtx*3+1],
        vtx2trgs[i_vtx*3+2] };
    for(int32_t idx = vtx2idx[i_vtx]; idx < vtx2idx[i_vtx+1]; ++idx ) {
        int32_t j_vtx = idx2vtx[idx];
        rhs[0] += lambda * vtx2vars_prev[j_vtx*3+0];
        rhs[1] += lambda * vtx2vars_prev[j_vtx*3+1];
        rhs[2] += lambda * vtx2vars_prev[j_vtx*3+2];
    }
    const float dtmp = 1.0 / (1.0 + float(vtx2idx[i_vtx+1] - vtx2idx[i_vtx]) * lambda);
    vtx2vars_next[i_vtx*3+0] = rhs[0] * dtmp;
    vtx2vars_next[i_vtx*3+1] = rhs[1] * dtmp;
    vtx2vars_next[i_vtx*3+2] = rhs[2] * dtmp;
}

__global__
void vtx2nvtx_from_uniform_mesh(
    const uint32_t num_vtx,
    const uint32_t *elem2vtx,
    const uint32_t num_node,
    const bool is_self,
    const uint32_t* vtx2jdx,
    const uint32_t* jdx2elem,
    uint32_t* vtx2nvtx,
    uint32_t* jdx2buff)
{
    const int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_vtx >= num_vtx) { return; }
    //
    const uint32_t jdx0 = vtx2jdx[i_vtx];
    const uint32_t jdx1 = vtx2jdx[i_vtx+1];
    const uint32_t m = is_self ? num_node : num_node - 1;
    uint32_t* buff = jdx2buff + jdx0 * m;
    vtx2nvtx[i_vtx] = 0;
    for(uint32_t jdx=jdx0;jdx<jdx1;++jdx){
        const uint32_t i_elem = jdx2elem[jdx];
        for(uint32_t i_node = 0; i_node < num_node; ++i_node) {
            const uint32_t j_vtx = elem2vtx[i_elem * num_node + i_node];
            if( j_vtx == i_vtx && !is_self ){ continue; }
            // check if j_vtx is already visited
            const uint32_t n = vtx2nvtx[i_vtx];
            bool is_visited = false;
            for(uint32_t i=0;i<n;++i){
               if( buff[i] == j_vtx ){ is_visited = true; break; }
            }
            if( is_visited ){ continue; }
            //
            buff[n] = j_vtx;
            vtx2nvtx[i_vtx] += 1;
        }
    }
}

__global__
void idx2vtx_from_vtx2buff_for_uniform_mesh(
    uint32_t num_vtx,
    const uint32_t* vtx2jdx,
    uint32_t num_node,
    const bool is_self,
    const uint32_t* vtx2idx,
    const uint32_t* jdx2buff,
    uint32_t* idx2vtx)
{
    const int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_vtx >= num_vtx) { return; }
    //
    const uint32_t m = is_self ? num_node : num_node - 1;
    const uint32_t jdx0 = vtx2jdx[i_vtx];
    const uint32_t* buff = jdx2buff + jdx0 * m;
    const uint32_t idx0 = vtx2idx[i_vtx];
    const uint32_t idx1 = vtx2idx[i_vtx + 1];
    for(uint32_t idx=idx0;idx<idx1;++idx){
        idx2vtx[idx] = buff[idx-idx0];
    }
}

__global__
void multiply_graph_laplacian(
    uint32_t num_vtx,
    const uint32_t* vtx2idx,
    const uint32_t* idx2vtx,
    uint32_t num_dim,
    const float* vtx2rhs,
    float* vtx2lhs)
{
    const int i_vtx = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_vtx >= num_vtx) { return; }
    // ---------
    const float valence = static_cast<float>(vtx2idx[i_vtx + 1] - vtx2idx[i_vtx]);
    float* lhs = vtx2lhs + i_vtx * num_dim;
    for(int i_dim=0;i_dim<num_dim;++i_dim) {
        lhs[i_dim] = valence * vtx2rhs[i_vtx * num_dim + i_dim];
    }
    for(int idx=vtx2idx[i_vtx];idx<vtx2idx[i_vtx + 1];++idx) {
        int j_vtx = idx2vtx[idx];
        for(int i_dim=0;i_dim<num_dim;++i_dim) {
            lhs[i_dim] -= vtx2rhs[j_vtx * num_dim + i_dim];
        }
    }
}

}

