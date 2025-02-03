//! functions related to c-style contiguous array of N dimensional coordinates

use num_traits::AsPrimitive;

pub fn cast<T, U>(vtx2xyz0: &[U]) -> Vec<T>
where
    T: Copy + 'static,
    U: AsPrimitive<T>,
{
    let res: Vec<T> = vtx2xyz0.iter().map(|v| v.as_()).collect();
    res
}

/*
pub fn from_array_of_nalgebra<T, const N: usize>(vtx2vecn: &Vec<nalgebra::SVector<T, N>>) -> Vec<T>
where
    T: nalgebra::RealField + Copy,
{
    let mut res = Vec::<T>::with_capacity(vtx2vecn.len() * N);
    for vec in vtx2vecn {
        res.extend(vec.iter());
    }
    res
}
 */

/*
pub fn to_array_of_nalgebra_vector<T, const N: usize>(vtx2xyz: &[T]) -> Vec<nalgebra::SVector<T, N>>
where
    T: nalgebra::RealField,
{
    vtx2xyz
        .chunks(N)
        .map(|v| nalgebra::SVector::<T, N>::from_row_slice(v))
        .collect()
}
 */

pub fn to_xn<T, const N: usize>(vtx2xyz: &[T], i_vtx: usize) -> &[T; N]
where
    T: num_traits::Float,
{
    vtx2xyz[i_vtx * N..i_vtx * N + N].try_into().unwrap()
}

pub fn cog<T, const N: usize>(vtx2xyz: &[T]) -> [T; N]
where
    T: num_traits::Float + Copy + 'static + std::iter::Sum<T>,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vecn::VecN;
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let mut cog = [T::zero(); N];
    for i_vtx in 0..num_vtx {
        let q0 = crate::vtx2xn::to_xn::<T, N>(vtx2xyz, i_vtx);
        cog.add_in_place(q0);
    }
    let s = T::one() / num_vtx.as_();
    cog.scale_in_place(s);
    cog
}

pub fn cov_cog<T, const N: usize>(vtx2xyz: &[T]) -> ([[T; N]; N], [T; N])
where
    T: num_traits::Float + Copy + 'static + std::iter::Sum,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vecn::VecN;
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let cog = cog::<T, N>(vtx2xyz);
    let mut cov = [[T::zero(); N]; N];
    for i_vtx in 0..num_vtx {
        let q = crate::vtx2xn::to_xn::<T, N>(vtx2xyz, i_vtx);
        let d = q.sub(&cog);
        for i in 0..N {
            for j in 0..N {
                cov[i][j] = cov[i][j] + d[i] * d[j];
            }
        }
    }
    (cov, cog)
}
