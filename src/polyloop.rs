//! methods for 2D and 3D poly loop

use num_traits::AsPrimitive;

/// return  arc-length of a 2D or 3D poly loop
pub fn arclength_vec<T, const N: usize>(
    vtxs: &Vec<nalgebra::SVector<T, N>>) -> T
    where T: nalgebra::RealField + Copy,
          f64: num_traits::AsPrimitive<T>
{
    if vtxs.len() < 2 { return T::zero(); }
    let np = vtxs.len();
    let mut len: T = T::zero();
    for ip0 in 0..np {
        let ip1 = (ip0 + 1) % np;
        len += (vtxs[ip0] - vtxs[ip1]).norm();
    }
    len
}

/// return  arc-length of a 2D or 3D poly loop
pub fn arclength<T, const N: usize>(
    vtx2xyz: &[T]) -> T
    where T: num_traits::Float + std::ops::AddAssign
{
    let np = vtx2xyz.len() / N;
    let mut len: T = T::zero();
    for ip0 in 0..np {
        let ip1 = (ip0 + 1) % np;
        let p0 = &vtx2xyz[ip0 * N..ip0 * N + N];
        let p1 = &vtx2xyz[ip1 * N..ip1 * N + N];
        len += del_geo::edge::length_::<T, N>(p0, p1);
    }
    len
}

pub fn edge2length<T, const N: usize>(
    vtx2xyz: &[T]) -> Vec<T>
    where T: num_traits::Float + std::ops::AddAssign
{
    let np = vtx2xyz.len() / N;
    let mut edge2length = Vec::<T>::with_capacity(np);
    for ip0 in 0..np {
        let ip1 = (ip0 + 1) % np;
        let p0 = &vtx2xyz[ip0 * N..ip0 * N + N];
        let p1 = &vtx2xyz[ip1 * N..ip1 * N + N];
        edge2length.push(del_geo::edge::length_::<T, N>(p0, p1));
    }
    edge2length
}

pub fn cog<T, const N: usize>(vtx2xyz: &[T]) -> nalgebra::SVector<T, N>
    where T: nalgebra::RealField + Copy + 'static,
          f64: AsPrimitive<T>
{
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vtx2xyz.len(), num_vtx * 3);
    let mut cog = nalgebra::SVector::<T, N>::zeros();
    let mut len = T::zero();
    for i_edge in 0..num_vtx {
        let iv0 = i_edge;
        let iv1 = (i_edge + 1) % num_vtx;
        let q0 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[iv0 * N..iv0 * N + N]);
        let q1 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[iv1 * N..iv1 * N + N]);
        let l = (q0 - q1).norm();
        cog += (q0 + q1).scale(0.5_f64.as_() * l);
        len += l;
    }
    cog / len
}

pub fn cov<T, const N: usize>(vtx2xyz: &[T]) -> nalgebra::SMatrix::<T, N, N>
    where T: nalgebra::RealField + Copy + 'static,
          f64: AsPrimitive<T>
{
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let cog = cog::<T, N>(vtx2xyz);
    let mut cov = nalgebra::SMatrix::<T, N, N>::zeros();
    for i_edge in 0..num_vtx {
        let iv0 = i_edge;
        let iv1 = (i_edge + 1) % num_vtx;
        let q0 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[iv0 * N..iv0 * N + N]) - cog;
        let q1 = nalgebra::SVector::<T, N>::from_row_slice(&vtx2xyz[iv1 * N..iv1 * N + N]) - cog;
        let l = (q0 - q1).norm();
        cov += (q0 * q0.transpose() + q1 * q1.transpose()).scale(l / 3_f64.as_());
        cov += (q0 * q1.transpose() + q1 * q0.transpose()).scale(l / 6_f64.as_());
    }
    cov
}

pub fn resample<T, const N: usize>(
    vtx2xyz_in: &[T],
    num_edge_out: usize) -> Vec<T>
    where T: nalgebra::RealField + num_traits::Float + Copy,
          f64: num_traits::AsPrimitive<T>,
          usize: num_traits::AsPrimitive<T>
{
    let mut v2x_out = Vec::<T>::new();
    let num_edge_in = vtx2xyz_in.len() / N;
    let len_edge_out = arclength::<T, N>(vtx2xyz_in) / num_edge_out.as_();
    v2x_out.extend_from_slice(&vtx2xyz_in[0..N]);
    let mut i_edge_in = 0;
    let mut traveled_ratio0 = T::zero();
    let mut remaining_length = len_edge_out;
    loop {
        if i_edge_in >= num_edge_in { break; }
        if v2x_out.len() >= num_edge_out * N { break; }
        let i0 = i_edge_in;
        let i1 = (i_edge_in + 1) % num_edge_in;
        let p0 = nalgebra::SVector::<T, N>::from_column_slice(&vtx2xyz_in[i0 * N..i0 * N + N]);
        let p1 = nalgebra::SVector::<T, N>::from_column_slice(&vtx2xyz_in[i1 * N..i1 * N + N]);
        let len_edge0 = (p1 - p0).norm();
        let len_togo0 = len_edge0 * (1_f64.as_() - traveled_ratio0);
        if len_togo0 > remaining_length { // put point in this segment
            traveled_ratio0 += remaining_length / len_edge0;
            let pn = p0.scale(1_f64.as_() - traveled_ratio0) + p1.scale(traveled_ratio0);
            v2x_out.extend(pn.iter());
            remaining_length = len_edge_out;
        } else { // next segment
            remaining_length -= len_togo0;
            traveled_ratio0 = 0_f64.as_();
            i_edge_in += 1;
        }
    }
    v2x_out
}