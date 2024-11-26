#pragma once

#include <cuda/std/array>

namespace mat2x3_col_major {

__device__
auto mult_vec3(
    const float* a,
    const float* b
) -> cuda::std::array<float,2>
{
    return {
        a[0] * b[0] + a[2] * b[1] + a[4] * b[2],
        a[1] * b[0] + a[3] * b[1] + a[5] * b[2],
    };
}

__device__
auto transform_ndc2pix(
    uint32_t img_w,
    uint32_t img_h
) -> cuda::std::array<float,6> {
    float img_w_f = float(img_w);
    float img_h_f = float(img_h);
    return {
        0.5f * img_w_f,
        0.f,
        0.f,
        -0.5f * img_h_f,
        0.5f * img_w_f,
        0.5f * img_h_f };
}

__device__
auto mult_mat3_col_major(
    const float* a,
    const float* b) -> cuda::std::array<float,6>
{
    return {
        a[0] * b[0] + a[2] * b[1] + a[4] * b[2],
        a[1] * b[0] + a[3] * b[1] + a[5] * b[2],
        a[0] * b[3] + a[2] * b[4] + a[4] * b[5],
        a[1] * b[3] + a[3] * b[4] + a[5] * b[5],
        a[0] * b[6] + a[2] * b[7] + a[4] * b[8],
        a[1] * b[6] + a[3] * b[7] + a[5] * b[8],
    };
}


}