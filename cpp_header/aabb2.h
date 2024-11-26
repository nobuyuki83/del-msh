#include <cuda/std/array>

namespace aabb2 {

__device__
auto from_point(
    const float* p,
    float rad) -> cuda::std::array<float,4>
{
    return {
        p[0] - rad,
        p[1] - rad,
        p[0] + rad,
        p[1] + rad};
}

__device__
auto scale(
    const float* aabb,
    float s
) -> cuda::std::array<float,4>
{
    return {
        (aabb[0] + aabb[2]) * 0.5f - (aabb[2] - aabb[0]) * 0.5f * s,
        (aabb[1] + aabb[3]) * 0.5f - (aabb[3] - aabb[1]) * 0.5f * s,
        (aabb[0] + aabb[2]) * 0.5f + (aabb[2] - aabb[0]) * 0.5f * s,
        (aabb[1] + aabb[3]) * 0.5f + (aabb[3] - aabb[1]) * 0.5f * s,
    };
}

__device__
auto translate(
    const float* aabb,
    const float* t
) -> cuda::std::array<float,4>
{
    return {
        aabb[0] + t[0],
        aabb[1] + t[1],
        aabb[2] + t[0],
        aabb[3] + t[1],
    };
}


__device__
bool is_inlcude_point(
    const float* aabb,
    const float* p)
{
    if( p[0] < aabb[0] ){
        return false;
    }
    if( p[1] < aabb[1] ){
        return false;
    }
    if( p[0] > aabb[2] ){
        return false;
    }
    if( p[1] > aabb[3] ){
        return false;
    }
    return true;
}


} // namespace aabb2