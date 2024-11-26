#pragma once

#include <cuda/std/array>

namespace quaternion {

__device__
auto to_mat3_col_major(
    const float* q) -> cuda::std::array<float,9>
{
    const float x2 = q[0] * q[0] * 2.f;
    const float y2 = q[1] * q[1] * 2.f;
    const float z2 = q[2] * q[2] * 2.f;
    const float xy = q[0] * q[1] * 2.f;
    const float yz = q[1] * q[2] * 2.f;
    const float zx = q[2] * q[0] * 2.f;
    const float xw = q[0] * q[3] * 2.f;
    const float yw = q[1] * q[3] * 2.f;
    const float zw = q[2] * q[3] * 2.f;
    return {
        1.f - y2 - z2,
        xy + zw,
        zx - yw,
        xy - zw,
        1.f - z2 - x2,
        yz + xw,
        zx + yw,
        yz - xw,
        1.f - x2 - y2,
    };
}

}