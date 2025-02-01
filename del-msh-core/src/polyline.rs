//! methods for polyline mesh

use num_traits::AsPrimitive;

pub fn cov<T, const N: usize>(vtx2xyz: &[T]) -> [[T; N]; N]
where
    T: num_traits::Float + Copy + 'static + std::iter::Sum,
    f64: AsPrimitive<T>,
{
    let one = T::one();
    let three = one + one + one;
    let six = three + three;
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let cog: [T; N] = crate::polyloop::cog_as_edges::<T, N>(vtx2xyz);
    let mut cov = [[T::zero(); N]; N];
    for i_edge in 0..num_vtx - 1 {
        let iv0 = i_edge;
        let iv1 = i_edge + 1;
        use del_geo_core::vecn::VecN;
        let q0: &[T; N] = &vtx2xyz[iv0 * N..iv0 * N + N].try_into().unwrap();
        let q0 = q0.sub(&cog);
        let q1: &[T; N] = &vtx2xyz[iv1 * N..iv1 * N + N].try_into().unwrap();
        let q1 = q1.sub(&cog);
        let l = q0.sub(&q1).norm();
        for i in 0..N {
            for j in 0..N {
                cov[i][j] = cov[i][j]
                    + (q0[i] * q0[j] + q1[i] * q1[j]) * (l / three)
                    + (q0[i] * q1[j] + q1[i] * q0[j]) * (l / six);
            }
        }
    }
    cov
}
