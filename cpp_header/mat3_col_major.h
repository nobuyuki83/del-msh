#include <cuda/std/optional>
#include <cuda/std/array>

namespace mat3_col_major {

__device__
auto mult_mat(const float* a, const float* b) -> cuda::std::array<float,9>
{
    cuda::std::array<float,9> o;
    for(int i=0;i<3;++i) {
        for(int j=0;j<3;++j) {
           o[i + j * 3] = 0.;
            for(int k=0;k<3;++k) {
                o[i + j * 3] += a[i + k * 3] * b[k + j * 3];
            }
        }
    }
    return o;
}

__device__
auto mult_vec(const float* a, const float* b) -> cuda::std::array<float,3>
{
    return  {
        a[0] * b[0] + a[3] * b[1] + a[6] * b[2],
        a[1] * b[0] + a[4] * b[1] + a[7] * b[2],
        a[2] * b[0] + a[5] * b[1] + a[8] * b[2]};
}

__device__
auto try_inverse(const float* b) -> cuda::std::optional< cuda::std::array<float,9> >
{
    const float det = b[0] * b[4] * b[8] + b[3] * b[7] * b[2] + b[6] * b[1] * b[5]
        - b[0] * b[7] * b[5]
        - b[6] * b[4] * b[2]
        - b[3] * b[1] * b[8];
    if( det == 0.f ){
        return {};
    }
    float inv_det = 1.f / det;
    return cuda::std::array<float,9>{
        inv_det * (b[4] * b[8] - b[5] * b[7]),
        inv_det * (b[2] * b[7] - b[1] * b[8]),
        inv_det * (b[1] * b[5] - b[2] * b[4]),
        inv_det * (b[5] * b[6] - b[3] * b[8]),
        inv_det * (b[0] * b[8] - b[2] * b[6]),
        inv_det * (b[2] * b[3] - b[0] * b[5]),
        inv_det * (b[3] * b[7] - b[4] * b[6]),
        inv_det * (b[1] * b[6] - b[0] * b[7]),
        inv_det * (b[0] * b[4] - b[1] * b[3]),
    };
}

}