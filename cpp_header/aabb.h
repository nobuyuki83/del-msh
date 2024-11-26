#pragma once

namespace aabb {

template<typename T>
__device__
constexpr auto min(T a, T b) { return a < b ? a : b; }

template<typename T>
__device__
constexpr auto max(T a, T b) { return a > b ? a : b; }


template <int ndim>
__device__
bool is_intersect_ray(
    const float* aabb,
    const float* ray_org,
    const float* ray_dir)
{
    float tmin = FLT_MIN;
    float tmax = FLT_MAX;
    for(int i_dim=0;i_dim<ndim;++i_dim) {
        if( fabs(ray_dir[i_dim]) !=0 ){
            const float t1 = (aabb[i_dim] - ray_org[i_dim]) / ray_dir[i_dim];
            const float t2 = (aabb[i_dim + ndim] - ray_org[i_dim]) / ray_dir[i_dim];
            tmin = max(tmin, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
        } else if( ray_org[i_dim] < aabb[i_dim] || ray_org[i_dim] > aabb[i_dim + ndim] ){
            return false;
        }
    }
    return tmax >= tmin && tmax >= 0.f;
}


}