//! methods for polyline mesh

use num_traits::AsPrimitive;

/// covariance matrix
pub fn cov<T, const N: usize>(vtx2xyz: &[T]) -> nalgebra::SMatrix<T, N, N>
where
    T: nalgebra::RealField + Copy + 'static,
    f64: AsPrimitive<T>,
{
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let cog = crate::polyloop::cog_as_edges::<T, N>(vtx2xyz);
    let mut cov = nalgebra::SMatrix::<T, N, N>::zeros();
    for i_edge in 0..num_vtx - 1 {
        let iv0 = i_edge;
        let iv1 = i_edge + 1;
        let q0 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[iv0 * N..iv0 * N + N]) - cog;
        let q1 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[iv1 * N..iv1 * N + N]) - cog;
        let l = (q0 - q1).norm();
        cov += (q0 * q0.transpose() + q1 * q1.transpose()).scale(l / 3_f64.as_());
        cov += (q0 * q1.transpose() + q1 * q0.transpose()).scale(l / 6_f64.as_());
    }
    cov
}
