#pragma once
#include <cuda/std/array>

namespace vec3 {

__device__
auto norm(const float* a) -> float
{
   const float sqnrm = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
   return sqrt(sqnrm);
}

__device__
auto add(const float* a, const float* b) -> cuda::std::array<float,3>
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

__device__
auto sub(const float* a, const float* b) -> cuda::std::array<float,3>
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

__device__
auto cross(const float* v1, const float* v2) -> cuda::std::array<float,3>
{
    return {
        v1[1] * v2[2] - v2[1] * v1[2],
        v1[2] * v2[0] - v2[2] * v1[0],
        v1[0] * v2[1] - v2[0] * v1[1],
    };
}

__device__
auto dot(const float* a, const float* b) -> float
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}




}