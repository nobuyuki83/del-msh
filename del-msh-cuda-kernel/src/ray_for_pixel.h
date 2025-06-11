#include "mat4_col_major.h"

__device__
auto ray_for_pixel(
    uint32_t i_pix,
    uint32_t img_w,
    uint32_t img_h,
    const float* transform_ndc2world) -> thrust::pair< cuda::std::array<float,3>, cuda::std::array<float,3>>
{
    using V3f = cuda::std::array<float,3>;
    const uint32_t i_h = i_pix / img_w;
    const uint32_t i_w = i_pix - i_h * img_w;
    const float x0 = 2.f * (float(i_w) + 0.5f) / float(img_w) - 1.f; // ndc_x
    const float y0 = 1.f - 2.f * (float(i_h) + 0.5f) / float(img_h); // ndc_y
    const auto q0 = mat4_col_major::transform_homogeneous(transform_ndc2world, V3f{x0, y0, +1.f}.data());
    const auto q1 = mat4_col_major::transform_homogeneous(transform_ndc2world, V3f{x0, y0, -1.f}.data());
    const auto v01 = vec3::sub(q1.data(), q0.data());
    return thrust::make_pair(q0,v01);
}