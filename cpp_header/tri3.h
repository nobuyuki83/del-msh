#pragma once
#include <cuda/std/optional>
#include "vec3.h"

namespace tri3 {

__device__
auto intersection_against_ray(
    const float *p0,
    const float *p1,
    const float *p2,
    const float *ray_org,
    const float *ray_dir) -> cuda::std::optional<float>
{
    using V3f = cuda::std::array<float,3>;
    float eps = 1.0e-5;
    const V3f edge1 = vec3::sub(p1, p0);
    const V3f edge2 = vec3::sub(p2, p0);
    const V3f pvec = vec3::cross(ray_dir, edge2.data());
    const float det = vec3::dot(edge1.data(), pvec.data());
    if( det > -eps && det < eps ){
        return {};
    }
    float invdet = 1.f / det;
    const V3f tvec = vec3::sub(ray_org, p0);
    float u = invdet * vec3::dot(tvec.data(), pvec.data());
    if( u < 0.f || u > 1.f ){
        return {};
    }
    const V3f qvec = vec3::cross(tvec.data(), edge1.data());
    const float v = invdet * vec3::dot(ray_dir, qvec.data());
    if( v < 0.f || u + v > 1.f ){
        return {};
    }
    // At this stage we can compute t to find out where the intersection point is on the line.
    const float t = invdet * vec3::dot(edge2.data(), qvec.data());
    return t;
}

}