use num_traits::AsPrimitive;

pub fn cast<T, U>(vtx2xyz0: &[U]) -> Vec<T>
where
    T: Copy + 'static,
    U: AsPrimitive<T>,
{
    let res: Vec<T> = vtx2xyz0.iter().map(|v| v.as_()).collect();
    res
}

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

pub fn to_array_of_nalgebra_vector<T, const N: usize>(vtx2xyz: &[T]) -> Vec<nalgebra::SVector<T, N>>
where
    T: nalgebra::RealField,
{
    vtx2xyz
        .chunks(N)
        .map(|v| nalgebra::SVector::<T, N>::from_row_slice(v))
        .collect()
}

pub fn cog<T, const N: usize>(vtx2xyz: &[T]) -> nalgebra::SVector<T, N>
    where
        T: nalgebra::RealField + Copy,
        usize: AsPrimitive<T>
{
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let mut cog = nalgebra::SVector::<T, N>::zeros();
    for i_vtx in 0..num_vtx {
        let q0 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[i_vtx * N..i_vtx * N + N]);
        cog += q0;
    }
    let s = T::one() / num_vtx.as_();
    cog *= s;
    cog
}

pub fn cov_cog<T, const N: usize>(vtx2xyz: &[T]) -> (nalgebra::SMatrix<T, N, N>, nalgebra::SVector<T, N>)
    where
        T: nalgebra::RealField + Copy,
        usize: AsPrimitive<T>
{
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let cog = cog::<T, N>(vtx2xyz);
    let mut cov = nalgebra::SMatrix::<T, N, N>::zeros();
    for i_vtx in 0..num_vtx {
        let q0 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[i_vtx * N..i_vtx * N + N]) - cog;
        cov += q0 * q0.transpose();
    }
    (cov,cog)
}