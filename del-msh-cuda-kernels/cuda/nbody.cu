#include "del_geo/vec3.h"

extern "C" {

__global__ void screened_poisson3(
    uint32_t num_wtx,
    const float* wtx2co,
    float* __restrict__ wtx2lhs,
    uint32_t num_vtx,
    const float* vtx2co,
    const float* __restrict__ vtx2rhs,
    float lambda,
    float eps
) {
    unsigned int i_wtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_wtx >= num_wtx) return;
    //
    float result[3] = {0.f, 0.f, 0.f};
    float sqrt_lambda = sqrtf(lambda);
    float norm = eps / __expf(-eps / sqrt_lambda);  // normalization factor
    //
    for (unsigned int j_vtx = 0; j_vtx < num_vtx; ++j_vtx) {
        auto r = vec3::sub(wtx2co + i_wtx * 3, vtx2co + j_vtx * 3);
        float r2 = vec3::dot(r.data(), r.data());
        float r_eps = sqrtf(r2 + eps * eps);
        float k = norm * __expf(-r_eps / sqrt_lambda) / r_eps;
        auto res = vec3::scale(vtx2rhs + j_vtx * 3, k);
        vec3::add_inplace(result, res.data());
    }
    wtx2lhs[i_wtx * 3 + 0] = result[0];
    wtx2lhs[i_wtx * 3 + 1] = result[1];
    wtx2lhs[i_wtx * 3 + 2] = result[2];
}

__global__
void elastic(
    unsigned int num_wtx,
    const float* __restrict__ wtx2co,
    float* __restrict__ wtx2lhs,
    uint32_t num_vtx,
    const float* vtx2co,
    const float* __restrict__ vtx2rhs,
    float nu,
    float eps
) {
    unsigned int i_wtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_wtx >= num_wtx) return;
    //
    const float PI = 3.14159265359f;
    const float a = 1.f / (4.f * PI);
    const float b = a / (4.f * (1.f - nu));
    const float norm = eps / (1.5f * a - b);

    float accum[3] = {0.f, 0.f, 0.f};

    const float* xi = &wtx2co[i_wtx * 3];

    for (unsigned int j_vtx = 0; j_vtx < num_vtx; ++j_vtx) {
        const float* xj = &vtx2co[j_vtx * 3];
        const float* gj = &vtx2rhs[j_vtx * 3];

        float r[3] = {
            xi[0] - xj[0],
            xi[1] - xj[1],
            xi[2] - xj[2]
        };

        float r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
        float r_eps = sqrtf(r2 + eps * eps);
        float r_eps_inv = 1.f / r_eps;
        float r_eps3_inv = 1.f / (r_eps * r_eps * r_eps);

        float coeff_i = norm * ((a - b) * r_eps_inv + 0.5f * a * eps * eps * r_eps3_inv);
        float coeff_rr_t = norm * b * r_eps3_inv;

        float dot_rg = r[0]*gj[0] + r[1]*gj[1] + r[2]*gj[2];

        accum[0] += coeff_i * gj[0] + coeff_rr_t * dot_rg * r[0];
        accum[1] += coeff_i * gj[1] + coeff_rr_t * dot_rg * r[1];
        accum[2] += coeff_i * gj[2] + coeff_rr_t * dot_rg * r[2];
    }

    wtx2lhs[i_wtx * 3 + 0] = accum[0];
    wtx2lhs[i_wtx * 3 + 1] = accum[1];
    wtx2lhs[i_wtx * 3 + 2] = accum[2];
}


}