
// for UINT_MAX
#include <limits.h>
//
#include "del_geo/vec3.h"
#include "del_geo/edge3.h"
#include "del_geo/mat4_col_major.h"


template <typename Model>
__device__ void filter_brute_force(
    uint32_t i_wtx,
    const float* wtx2xyz,
    float* __restrict__ wtx2lhs,
    uint32_t num_vtx,
    const float* vtx2xyz,
    const float* __restrict__ vtx2rhs,
    Model model
) {
    const float* pos_i = wtx2xyz + i_wtx * 3;
    float lhs_i[3] = {0.f};
    for (uint32_t j_vtx = 0; j_vtx < num_vtx; ++j_vtx) {
        const auto rel_pos = vec3::sub(pos_i, vtx2xyz + j_vtx * 3);
        const auto res = model.eval(rel_pos.data(), vtx2rhs + j_vtx * 3);
        vec3::add_inplace(lhs_i, res.data());
    }
    wtx2lhs[i_wtx * 3 + 0] = lhs_i[0];
    wtx2lhs[i_wtx * 3 + 1] = lhs_i[1];
    wtx2lhs[i_wtx * 3 + 2] = lhs_i[2];
}

template <typename Model>
__device__ void filter_with_acceleration(
    uint32_t i_wtx,
    const float* wtx2xyz,
    float* __restrict__ wtx2lhs,
    uint32_t num_vtx,
    const float* vtx2xyz,
    const float* __restrict__ vtx2rhs,
    Model model,
    const float* transform_world2unit,
    const float* transform_unit2world,
    const uint32_t num_onode,
    uint32_t* onode2idx_tree,
    const float* onode2center,
    const uint32_t* onode2depth,
    const uint32_t* idx2jdx_offset,
    const uint32_t* jdx2vtx,
    const float* onode2gcunit,
    const float* onode2rhs,
    const float theta
) {
    const float pos_i_world[3] = {
        wtx2xyz[i_wtx*3],
        wtx2xyz[i_wtx*3+1],
        wtx2xyz[i_wtx*3+2] };
    const auto pos_i_unit = mat4_col_major::transform_homogeneous(transform_world2unit, pos_i_world);
    //
    float lhs[3] = {0.f};
    //
    uint32_t stack_node[56];
    int stack_top = 0;
    stack_node[stack_top++] = 0; // root node

    while(stack_top>0) {
        const int j_onode = stack_node[--stack_top];
        const uint32_t depth = onode2depth[j_onode];

        const float* cg_unit = onode2gcunit + j_onode * 3;
        const float* center_unit = onode2center + j_onode * 3;
        const float delta_unit = edge3::length(center_unit, cg_unit);

        const auto rel_unit = vec3::sub(pos_i_unit.data(), cg_unit);
        const float dist_unit = vec3::norm(rel_unit.data());
        const float celllen_unit = 1.0 / static_cast<float>(1 << depth);
        //
        const bool is_far = dist_unit - delta_unit > 0. && celllen_unit < (dist_unit - delta_unit) * theta;
        if( is_far ){
            const auto cg_world = mat4_col_major::transform_homogeneous(transform_unit2world, cg_unit);
            const auto rel_world = vec3::sub(pos_i_world, cg_world.data());
            const auto res = model.eval(rel_world.data(), onode2rhs + j_onode * 3);
            vec3::add_inplace(lhs, res.data());
        }
        else{
            for(int j_child=0;j_child<8;++j_child){
                uint32_t j = onode2idx_tree[j_onode*9 + 1 + j_child];
                if( j == UINT_MAX ){ continue; }
                if( j < num_onode ){
                    stack_node[stack_top++] = j;
                }
                else{
                    int j_idx = j - num_onode;
                    for(int jdx=idx2jdx_offset[j_idx];jdx<idx2jdx_offset[j_idx+1];++jdx){
                        int j_vtx = jdx2vtx[jdx];
                        const float* pos_j_world = vtx2xyz + j_vtx * 3;
                        const auto rel_world = vec3::sub(pos_i_world, pos_j_world);
                        const auto res = model.eval(rel_world.data(), vtx2rhs + j_vtx * 3);
                        vec3::add_inplace(lhs, res.data());
                    }
                }
            }
        }
    }
    wtx2lhs[i_wtx*3+0] = lhs[0];
    wtx2lhs[i_wtx*3+1] = lhs[1];
    wtx2lhs[i_wtx*3+2] = lhs[2];
}


extern "C" {

class ScreenedPoisson3 {   
public:
    float sqrt_lambda;
    float eps;
    float norm;
public:
    __device__
    ScreenedPoisson3(float lambda_, float eps_) {        
        this->sqrt_lambda = sqrtf(lambda_);
        this->norm = eps_ / __expf(-eps_ / sqrt_lambda);  // normalization factor
        this->eps = eps_;
    }

    __device__
    auto eval(const float* rel_pos, const float* rhs_j) const -> cuda::std::array<float,3> {
        const float r2 = vec3::dot(rel_pos, rel_pos);
        const float r_eps = sqrtf(r2 + eps * eps);
        const float k = norm * __expf(-r_eps / sqrt_lambda) / r_eps;
        return vec3::scale(rhs_j, k);
    }
};

class Elastic3 {
public:
    float a;
    float b;
    float eps;
    float norm;
public:

    __device__
    Elastic3(float nu_, float eps_) {
        const float PI = 3.14159265359f;
        this->a = 1.f / (4.f * PI);
        this->b = this->a / (4.f * (1.f - nu_));  
        this->eps = eps_;
        this->norm = eps_ / (1.5f * this->a - this->b);
    }

    __device__
    auto eval(const float* rel_pos, const float* rhs_j) const -> cuda::std::array<float,3> {        
        const float r2 = vec3::dot(rel_pos, rel_pos);
        const float r_eps = sqrtf(r2 + eps * eps);
        const float r_eps_inv = 1.f / r_eps;
        const float r_eps3_inv = 1.f / (r_eps * r_eps * r_eps);
        const float coeff_i = this->norm * ((this->a - this->b) * r_eps_inv + 0.5f * this->a * this->eps * this->eps * r_eps3_inv);
        const float coeff_rr_t = this->norm * this->b * r_eps3_inv;
        const float dot_rg = vec3::dot(rel_pos, rhs_j);
        return {
            coeff_i * rhs_j[0] + coeff_rr_t * dot_rg * rel_pos[0],
            coeff_i * rhs_j[1] + coeff_rr_t * dot_rg * rel_pos[1],
            coeff_i * rhs_j[2] + coeff_rr_t * dot_rg * rel_pos[2]
        };
    }
};

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
    const auto model = ScreenedPoisson3(lambda, eps);
    filter_brute_force(
        i_wtx,
        wtx2co,
        wtx2lhs,
        num_vtx,
        vtx2co,
        vtx2rhs,
        model
    );
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
    const auto elastic = Elastic3(nu, eps);
    filter_brute_force(
        i_wtx,
        wtx2co,
        wtx2lhs,
        num_vtx,
        vtx2co,
        vtx2rhs,
        elastic
    );
}


__global__ void screened_poisson3_with_acceleration(
    uint32_t num_wtx,
    const float* wtx2xyz,
    float* __restrict__ wtx2lhs,
    uint32_t num_vtx,
    const float* vtx2xyz,
    const float* __restrict__ vtx2rhs,
    float lambda,
    float eps,
    const float* transform_world2unit,
    const float* transform_unit2world,
    const uint32_t num_onode,
    uint32_t* onode2idx_tree,
    const float* onode2center,
    const uint32_t* onode2depth,
    const uint32_t* idx2jdx_offset,
    const uint32_t* jdx2vtx,
    const float* onode2gcunit,
    const float* onode2rhs,
    const float theta
) {
    unsigned int i_wtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_wtx >= num_wtx) return;
    // ---------
    const auto model = ScreenedPoisson3(lambda, eps);
    filter_with_acceleration(
        i_wtx,
        wtx2xyz,
        wtx2lhs,
        num_vtx,
        vtx2xyz,
        vtx2rhs,
        model,
        transform_world2unit,
        transform_unit2world,
        num_onode,
        onode2idx_tree,
        onode2center,
        onode2depth,
        idx2jdx_offset,
        jdx2vtx,
        onode2gcunit,
        onode2rhs,
        theta
    );  
}


__global__ void elastic3_with_acceleration(
    uint32_t num_wtx,
    const float* wtx2xyz,
    float* __restrict__ wtx2lhs,
    uint32_t num_vtx,
    const float* vtx2xyz,
    const float* __restrict__ vtx2rhs,
    float nu,
    float eps,
    const float* transform_world2unit,
    const float* transform_unit2world,
    const uint32_t num_onode,
    uint32_t* onode2idx_tree,
    const float* onode2center,
    const uint32_t* onode2depth,
    const uint32_t* idx2jdx_offset,
    const uint32_t* jdx2vtx,
    const float* onode2gcunit,
    const float* onode2rhs,
    const float theta
) {
    unsigned int i_wtx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_wtx >= num_wtx) return;
    // ---------
    const auto model = Elastic3(nu, eps);
    filter_with_acceleration(
        i_wtx,
        wtx2xyz,
        wtx2lhs,
        num_vtx,
        vtx2xyz,
        vtx2rhs,
        model,
        transform_world2unit,
        transform_unit2world,
        num_onode,
        onode2idx_tree,
        onode2center,
        onode2depth,
        idx2jdx_offset,
        jdx2vtx,
        onode2gcunit,
        onode2rhs,
        theta
    );  
}


}