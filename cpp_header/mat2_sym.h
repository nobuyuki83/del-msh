#include <cuda/std/array>
#include <cuda/std/tuple>
#include <float.h>
//
#include "mat2x3_col_major.h"
#include "quaternion.h"

namespace mat2_sym {

__device__
auto projected_spd_mat3(
    const float* p_mat,
    const float* quat0,
    const float* d) -> cuda::std::array<float,3>
{
    const cuda::std::array<float,9> r = quaternion::to_mat3_col_major(quat0);
    const cuda::std::array<float,6> pr = mat2x3_col_major::mult_mat3_col_major(p_mat, r.data());
    const float dd[3] = {d[0] * d[0], d[1] * d[1], d[2] * d[2]};
    return {
        pr[0] * pr[0] * dd[0] + pr[2] * pr[2] * dd[1] + pr[4] * pr[4] * dd[2],
        pr[0] * pr[1] * dd[0] + pr[2] * pr[3] * dd[1] + pr[4] * pr[5] * dd[2],
        pr[1] * pr[1] * dd[0] + pr[3] * pr[3] * dd[1] + pr[5] * pr[5] * dd[2],
    };
}

/// ax^2 + 2bxy + cy^2 = 1
__device__
auto principal_directions(
    const float* coeff
) -> cuda::std::tuple< cuda::std::array<float,2>, cuda::std::array<cuda::std::array<float,2>,2>>
{
    const float a = coeff[0];
    const float b = coeff[1];
    const float c = coeff[2];
    if( b == 0.f ){
        const cuda::std::array<float,2> v0 = {1.f, 0.f};
        const cuda::std::array<float,2> v1 = {0.f, 1.f};
        return {{a,c}, {v0, v1}};
    }
    float tmp = sqrt((a - c) * (a - c) + 4.f * b * b);
    float lam0 = 0.5f * (a + c - tmp);
    float lam1 = 0.5f * (a + c + tmp);
    float det0 = a - c + tmp;
    float det1 = a - c - tmp;
    if( fabs(det0) > fabs(det1) ){
        const cuda::std::array<float,2> v0 = {-2.f * b, det0};
        const cuda::std::array<float,2> v1 = {det0, 2.f * b};
        return {{lam0, lam1}, {v0, v1}};
    } else {
        const cuda::std::array<float,2> v0 = {det1, 2.f * b};
        const cuda::std::array<float,2> v1 = {-2.f * b, det1};
        return {{lam0, lam1}, {v0, v1}};
    }
}

__device__
auto safe_inverse(
    const float* coeff
) -> cuda::std::array<float,3>
{
    const float a = coeff[0];
    const float b = coeff[1];
    const float c = coeff[2];
    const float det = a * c - b * b;
    if( det <= FLT_EPSILON ){
        float l = 1.f / FLT_EPSILON;
        float a1 = a + FLT_EPSILON;
        float c1 = c + FLT_EPSILON;
        return {c1 * l, -b * l, a1 * l};
    }
    float di = 1.f / det;
    return {di * c, -di * b, di * a};
}

__device__
auto safe_inverse_preserve_positive_definiteness(
    const float* abc,
    float eps
) -> cuda::std::array<float,3>
{
    const float eig_min = (abc[0] + abc[2]) * eps;
    if( fabs(abc[0] * abc[2] - abc[1] * abc[1]) < eig_min ){
        // one of the eigen value is nearly zero
        const auto eigen = principal_directions(abc);
        const auto e = cuda::std::get<0>(eigen);
        const auto v = cuda::std::get<1>(eigen);
        float e0inv = 1.f / (e[0] + eps);
        float e1inv = 1.f / (e[1] + eps);
        return {
            e0inv * v[0][0] * v[0][0] + e1inv * v[1][0] * v[1][0],
            e0inv * v[0][1] * v[0][0] + e1inv * v[1][1] * v[1][0],
            e0inv * v[0][1] * v[0][1] + e1inv * v[1][1] * v[1][1],
        };
    } else {
        return safe_inverse(abc);
    }
}

__device__
auto aabb2(
    const float* coeff
) -> cuda::std::array<float,4>
{
    const float a = coeff[0];
    const float b = coeff[1];
    const float c = coeff[2];
    const float det = a * c - b * b;
    const float minx = sqrt(c / det);
    const float miny = sqrt(a / det);
    return {-minx, -miny, minx, miny};
}


__device__
float mult_vec_from_both_sides(
    const float* m,
    const float* b,
    const float* c)
{
    return m[0] * b[0] * c[0] + m[1] * (b[0] * c[1] + b[1] * c[0]) + m[2] * b[1] * c[1];
}


} // namespace mat2_sym