#pragma once

#include <cuda/std/array>

namespace aabb3 {

__device__
auto from_point(
    const float* p,
    float rad) -> cuda::std::array<float,6>
{
    return {
        p[0] - rad,
        p[1] - rad,
        p[2] - rad,
        p[0] + rad,
        p[1] + rad,
        p[2] + rad
    };
}

__device__
auto set_point(
    float* aabb,
    const float* xyz,
    float rad)
{
    aabb[0] = xyz[0] - rad;
    aabb[1] = xyz[1] - rad;
    aabb[2] = xyz[2] - rad;
    aabb[3] = xyz[0] + rad;
    aabb[4] = xyz[1] + rad;
    aabb[5] = xyz[2] + rad;
}

__device__
auto add_point(
    float* aabb,
    const float* xyz,
    float eps)
{
    aabb[0] = min(aabb[0], xyz[0] - eps);
    aabb[3] = max(aabb[3], xyz[0] + eps);
    aabb[1] = min(aabb[1], xyz[1] - eps);
    aabb[4] = max(aabb[4], xyz[1] + eps);
    aabb[2] = min(aabb[2], xyz[2] - eps);
    aabb[5] = max(aabb[5], xyz[2] + eps);
}

__device__
auto set_merged_two_aabbs(
    float* o,
    const float* i0,
    const float* i1)
{
    for (int i=0;i<3;++i) {
        o[i] = (i0[i] < i1[i]) ? i0[i] : i1[i];
        o[i + 3] = (i0[i + 3] > i1[i + 3]) ? i0[i + 3] : i1[i + 3];
    }
}


}