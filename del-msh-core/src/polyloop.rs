//! methods for 2D and 3D poly loop

use num_traits::AsPrimitive;

/// return  arc-length of a 2D or 3D poly loop
pub fn arclength_from_vtx2vecn<T, const N: usize>(vtxs: &[nalgebra::SVector<T, N>]) -> T
where
    T: nalgebra::RealField + Copy,
    f64: AsPrimitive<T>,
{
    if vtxs.len() < 2 {
        return T::zero();
    }
    let np = vtxs.len();
    let mut len: T = T::zero();
    for ip0 in 0..np {
        let ip1 = (ip0 + 1) % np;
        len += (vtxs[ip0] - vtxs[ip1]).norm();
    }
    len
}

/// return  arc-length of a 2D or 3D poly loop
pub fn arclength<T, const N: usize>(vtx2xyz: &[T]) -> T
where
    T: num_traits::Float + std::ops::AddAssign,
{
    let np = vtx2xyz.len() / N;
    let mut len: T = T::zero();
    for ip0 in 0..np {
        let ip1 = (ip0 + 1) % np;
        let p0 = &vtx2xyz[ip0 * N..ip0 * N + N];
        let p1 = &vtx2xyz[ip1 * N..ip1 * N + N];
        len += del_geo_core::edge::length::<T, N>(p0, p1);
    }
    len
}

pub fn edge2length<T, const N: usize>(vtx2xyz: &[T]) -> Vec<T>
where
    T: num_traits::Float + std::ops::AddAssign,
{
    let np = vtx2xyz.len() / N;
    let mut edge2length = Vec::<T>::with_capacity(np);
    for ip0 in 0..np {
        let ip1 = (ip0 + 1) % np;
        let p0 = &vtx2xyz[ip0 * N..ip0 * N + N];
        let p1 = &vtx2xyz[ip1 * N..ip1 * N + N];
        edge2length.push(del_geo_core::edge::length::<T, N>(p0, p1));
    }
    edge2length
}

/// the center of gravity for polyloop.
/// Here polyloop is a looped wire, not the polygonal face bounded by the polyloop
pub fn cog_as_edges<T, const N: usize>(vtx2xyz: &[T]) -> nalgebra::SVector<T, N>
where
    T: nalgebra::RealField + Copy + 'static,
    f64: AsPrimitive<T>,
{
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
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

#[test]
fn test_cog() {
    let mut vtx2xy = crate::polyloop2::from_circle(1f32, 32);
    let (x0, y0) = (1.3, 0.5);
    vtx2xy
        .view_mut((0, 0), (1, vtx2xy.ncols()))
        .add_scalar_mut(x0);
    vtx2xy
        .view_mut((1, 0), (1, vtx2xy.ncols()))
        .add_scalar_mut(y0);
    let cog = cog_as_edges::<f32, 2>(vtx2xy.as_slice());
    assert!(
        del_geo_core::edge::length::<f32, 2>(&[x0, y0], cog.as_slice().try_into().unwrap())
            < 1.0e-5
    );
}

/// the center of gravity for polyloop.
/// Here "polyloop" is a looped wire, not the polygonal face bounded by the polyloop
pub fn cog_from_vtx2vecn_as_edges<T, const N: usize>(
    vtx2xyz: &[nalgebra::SVector<T, N>],
) -> nalgebra::SVector<T, N>
where
    T: nalgebra::RealField + Copy + 'static,
    f64: AsPrimitive<T>,
{
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vtx2xyz.len(), num_vtx);
    let mut cog = nalgebra::SVector::<T, N>::zeros();
    let mut len = T::zero();
    for i_edge in 0..num_vtx {
        let iv0 = i_edge;
        let iv1 = (i_edge + 1) % num_vtx;
        let q0 = &vtx2xyz[iv0];
        let q1 = &vtx2xyz[iv1];
        let l = (q0 - q1).norm();
        cog += (q0 + q1).scale(0.5_f64.as_() * l);
        len += l;
    }
    cog / len
}

pub fn cov<T, const N: usize>(vtx2xyz: &[T]) -> nalgebra::SMatrix<T, N, N>
where
    T: nalgebra::RealField + Copy + 'static,
    f64: AsPrimitive<T>,
{
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let cog = cog_as_edges::<T, N>(vtx2xyz);
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

pub fn resample<T, const N: usize>(vtx2xyz_in: &[T], num_edge_out: usize) -> Vec<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let mut v2x_out = Vec::<T>::new();
    let num_edge_in = vtx2xyz_in.len() / N;
    let len_edge_out = arclength::<T, N>(vtx2xyz_in) / num_edge_out.as_();
    v2x_out.extend_from_slice(&vtx2xyz_in[0..N]);
    let mut i_edge_in = 0;
    let mut traveled_ratio0 = T::zero();
    let mut remaining_length = len_edge_out;
    loop {
        if i_edge_in >= num_edge_in {
            break;
        }
        if v2x_out.len() >= num_edge_out * N {
            break;
        }
        let i0 = i_edge_in;
        let i1 = (i_edge_in + 1) % num_edge_in;
        let p0 = nalgebra::SVector::<T, N>::from_column_slice(&vtx2xyz_in[i0 * N..i0 * N + N]);
        let p1 = nalgebra::SVector::<T, N>::from_column_slice(&vtx2xyz_in[i1 * N..i1 * N + N]);
        let len_edge0 = (p1 - p0).norm();
        let len_togo0 = len_edge0 * (1_f64.as_() - traveled_ratio0);
        if len_togo0 > remaining_length {
            // put point in this segment
            traveled_ratio0 += remaining_length / len_edge0;
            let pn = p0.scale(1_f64.as_() - traveled_ratio0) + p1.scale(traveled_ratio0);
            v2x_out.extend(pn.iter());
            remaining_length = len_edge_out;
        } else {
            // next segment
            remaining_length -= len_togo0;
            traveled_ratio0 = 0_f64.as_();
            i_edge_in += 1;
        }
    }
    v2x_out
}

#[allow(clippy::identity_op)]
pub fn resample_multiple_loops_remain_original_vtxs<T>(
    loop2idx_inout: &mut Vec<usize>,
    idx2vtx_inout: &mut Vec<usize>,
    vtx2vec_inout: &mut Vec<nalgebra::Vector2<T>>,
    max_edge_length: T,
) where
    T: nalgebra::RealField + Copy + AsPrimitive<usize>,
    usize: AsPrimitive<T>,
{
    assert_eq!(vtx2vec_inout.len(), idx2vtx_inout.len());
    let loop2idx_in = loop2idx_inout.clone();
    let idx2vtx_in = idx2vtx_inout.clone();
    assert!(idx2vtx_in.len() >= 2);
    let num_loop = loop2idx_in.len() - 1;
    let mut edge2point: Vec<Vec<usize>> = vec![vec!(); idx2vtx_in.len()];
    {
        for i_loop in 0..num_loop {
            assert!(loop2idx_in[i_loop + 1] > loop2idx_in[i_loop]);
            let np = loop2idx_in[i_loop + 1] - loop2idx_in[i_loop];
            for ip in 0..np {
                let iipo0 = loop2idx_in[i_loop] + (ip + 0) % np;
                let iipo1 = loop2idx_in[i_loop] + (ip + 1) % np;
                assert!(iipo0 < idx2vtx_in.len());
                assert!(iipo1 < idx2vtx_in.len());
                let ipo0 = idx2vtx_in[iipo0];
                let ipo1 = idx2vtx_in[iipo1];
                assert!(ipo0 < vtx2vec_inout.len());
                assert!(ipo1 < vtx2vec_inout.len());
                let po0 = vtx2vec_inout[ipo0]; // never use reference here because aVec2 will resize afterward
                let po1 = vtx2vec_inout[ipo1]; // never use reference here because aVec2 will resize afterward
                let nadd: usize = ((po0 - po1).norm() / max_edge_length).as_();
                if nadd == 0 {
                    continue;
                }
                for iadd in 0..nadd {
                    let r2: T = (iadd + 1).as_() / (nadd + 1).as_();
                    let v2 = po0.scale(T::one() - r2) + po1.scale(r2);
                    let ipo2 = vtx2vec_inout.len();
                    vtx2vec_inout.push(v2);
                    assert!(iipo0 < edge2point.len());
                    edge2point[iipo0].push(ipo2);
                }
            }
        }
    }
    ////
    loop2idx_inout.resize(num_loop + 1, usize::MAX);
    loop2idx_inout[0] = 0;
    for iloop in 0..num_loop {
        let nbar0 = loop2idx_in[iloop + 1] - loop2idx_in[iloop];
        let mut nbar1 = nbar0;
        for ibar in 0..nbar0 {
            let iip_loop = loop2idx_in[iloop] + ibar;
            nbar1 += edge2point[iip_loop].len();
        }
        loop2idx_inout[iloop + 1] = loop2idx_inout[iloop] + nbar1;
    }
    // adding new vertices on the outline
    idx2vtx_inout.resize(loop2idx_inout[num_loop], usize::MAX);
    let mut i_vtx0 = 0;
    for i_loop in 0..num_loop {
        for iip_loop in loop2idx_in[i_loop]..loop2idx_in[i_loop + 1] {
            let ip_loop = idx2vtx_in[iip_loop];
            idx2vtx_inout[i_vtx0] = ip_loop;
            i_vtx0 += 1;
            for iadd in 0..edge2point[ip_loop].len() {
                idx2vtx_inout[i_vtx0] = edge2point[iip_loop][iadd];
                i_vtx0 += 1;
            }
        }
    }
    assert_eq!(idx2vtx_inout.len(), vtx2vec_inout.len());
    assert_eq!(idx2vtx_inout.len(), i_vtx0);
}

pub fn to_cylinder_trimeshes<Real>(
    vtx2xy: &[Real],
    num_dim: usize,
    radius: Real,
) -> (Vec<usize>, Vec<Real>)
where
    Real: nalgebra::RealField + num_traits::FloatConst + 'static + Copy,
    usize: AsPrimitive<Real>,
    f64: AsPrimitive<Real>,
{
    let num_vtx = vtx2xy.len() / num_dim;
    let mut out_tri2vtx: Vec<usize> = vec![];
    let mut out_vtx2xyz: Vec<Real> = vec![];
    for i_edge in 0..num_vtx {
        let i0 = i_edge;
        let i1 = (i_edge + 1) % num_vtx;
        let p0 = &vtx2xy[i0 * num_dim..(i0 + 1) * num_dim];
        let p1 = &vtx2xy[i1 * num_dim..(i1 + 1) * num_dim];
        let p0 = if num_dim == 3 {
            nalgebra::Vector3::<Real>::new(p0[0], p0[1], p0[2])
        } else {
            nalgebra::Vector3::<Real>::new(p0[0], p0[1], Real::zero())
        };
        let p1 = if num_dim == 3 {
            nalgebra::Vector3::<Real>::new(p1[0], p1[1], p1[2])
        } else {
            nalgebra::Vector3::<Real>::new(p1[0], p1[1], Real::zero())
        };
        let (tri2vtx, vtx2xyz) =
            crate::trimesh3_primitive::cylinder_open_connecting_two_points(32, radius, p0, p1);
        crate::uniform_mesh::merge(
            &mut out_tri2vtx,
            &mut out_vtx2xyz,
            tri2vtx.as_slice(),
            vtx2xyz.as_slice(),
            3,
        );
    }
    (out_tri2vtx, out_vtx2xyz)
}

pub fn edge2vtx(num_vtx: usize) -> Vec<usize> {
    let mut edge2vtx = Vec::<usize>::with_capacity(num_vtx * 2);
    for i_vtx in 0..num_vtx {
        edge2vtx.push(i_vtx);
        edge2vtx.push((i_vtx + 1) % num_vtx);
    }
    edge2vtx
}
