#include <cstdint>
#include <cuda_runtime.h>

extern "C" {

__global__
void from_polygon_mesh_as_points(
    const uint32_t num_elem,
    const uint32_t *elem2idx_offset,
    const uint32_t *idx2vtx,
    const float *vtx2xyz,
    const uint32_t num_dim,
    float *elem2cog)
{
    const uint32_t i_elem = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_elem >= num_elem) { return; }

    const uint32_t i_idx_begin = elem2idx_offset[i_elem];
    const uint32_t i_idx_end   = elem2idx_offset[i_elem + 1];
    const uint32_t num_vtx_in_elem = i_idx_end - i_idx_begin;

    float* cog = elem2cog + i_elem * num_dim;
    for (uint32_t idim = 0; idim < num_dim; ++idim) {
        cog[idim] = 0.f;
    }

    for (uint32_t i_idx = i_idx_begin; i_idx < i_idx_end; ++i_idx) {
        const uint32_t i_vtx = idx2vtx[i_idx];
        for (uint32_t idim = 0; idim < num_dim; ++idim) {
            cog[idim] += vtx2xyz[i_vtx * num_dim + idim];
        }
    }

    if (num_vtx_in_elem > 0) {
        const float ratio = 1.f / static_cast<float>(num_vtx_in_elem);
        for (uint32_t idim = 0; idim < num_dim; ++idim) {
            cog[idim] *= ratio;
        }
    }
}

} // extern "C"
