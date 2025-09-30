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

}

