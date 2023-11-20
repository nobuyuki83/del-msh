//! methods that generate meshes of primitive shapes (e.g., cylinder, torus)

use num_traits::AsPrimitive;

/// generate 3D mesh of closed cylinder
/// * `r` - radius
/// * 'l' - length
pub fn cylinder_closed_end_tri3<T>(
    r: T,
    l: T,
    nr: usize,
    nl: usize) -> (Vec<usize>, Vec<T>)
    where T: num_traits::Float + 'static,
          f32: num_traits::AsPrimitive<T>,
          usize: num_traits::AsPrimitive<T>
{
    let mut vtx2xyz = Vec::<T>::new();
    let mut tri2vtx = Vec::<usize>::new();
    if nl < 1 || nr <= 2 {
        return (tri2vtx, vtx2xyz);
    }
    let pi: T = std::f32::consts::PI.as_();
    let dl: T = l / (nl).as_();
    let dr: T = (2.).as_() * pi / (nr).as_();
    vtx2xyz.reserve((nr * (nl + 1) + 2) * 3);
    {
        vtx2xyz.push(0.as_());
        vtx2xyz.push(-l * (0.5).as_());
        vtx2xyz.push((0.).as_());
    }
    for il in 0..nl + 1 {
        let y0 = -l * (0.5).as_() + dl * (il).as_();
        for ilo in 0..nr {
            let x0 = r * (dr * (ilo).as_()).cos();
            let z0 = r * (dr * (ilo).as_()).sin();
            vtx2xyz.push(x0);
            vtx2xyz.push(y0);
            vtx2xyz.push(z0);
        }
    }
    {
        vtx2xyz.push((0.).as_());
        vtx2xyz.push(l * (0.5).as_());
        vtx2xyz.push((0.).as_());
    }
    // ------------------------------------
    let nla = nl + 2;
    let ntri = nr * (nla - 1) * 2 + nr * 2;
    tri2vtx.reserve(ntri * 3);
    for ilo in 0..nr {
        tri2vtx.push(0);
        tri2vtx.push((ilo + 0) % nr + 1);
        tri2vtx.push((ilo + 1) % nr + 1);
    }
    for ila in 0..nla - 2 {
        for ilo in 0..nr {
            let i1 = (ila + 0) * nr + 1 + (ilo + 0) % nr;
            let i2 = (ila + 0) * nr + 1 + (ilo + 1) % nr;
            let i3 = (ila + 1) * nr + 1 + (ilo + 1) % nr;
            let i4 = (ila + 1) * nr + 1 + (ilo + 0) % nr;
            tri2vtx.push(i3);
            tri2vtx.push(i2);
            tri2vtx.push(i1);
            tri2vtx.push(i4);
            tri2vtx.push(i3);
            tri2vtx.push(i1);
        }
    }
    for ilo in 0..nr {
        tri2vtx.push(nr * (nla - 1) + 1);
        tri2vtx.push((nla - 2) * nr + 1 + (ilo + 1) % nr);
        tri2vtx.push((nla - 2) * nr + 1 + (ilo + 0) % nr);
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_cylider_closed_end_tri3() {
    cylinder_closed_end_tri3::<f32>(1., 1., 32, 32);
    cylinder_closed_end_tri3::<f64>(1., 1., 32, 32);
}

// ------------------------

pub fn capsule_tri3<T>(
    r: T,
    l: T,
    nc: usize,
    nr: usize,
    nl: usize) -> (Vec<usize>, Vec<T>)
    where T: num_traits::Float + 'static,
          f32: num_traits::AsPrimitive<T>,
          usize: num_traits::AsPrimitive<T>
{
    let (tri2vtx, mut vtx2xyz) = cylinder_closed_end_tri3::<T>(
        (1.).as_(), (1.).as_(),
        nc, 2 * nr + nl - 2);
    assert_eq!(vtx2xyz.len() / 3, (2 * nr + nl - 1) * nc + 2);
    let pi: T = (std::f32::consts::PI).as_();
    {
        vtx2xyz[0 * 3 + 0] = 0.as_();
        vtx2xyz[0 * 3 + 1] = -l * 0.5.as_() - r;
        vtx2xyz[0 * 3 + 2] = 0.as_();
    }
    for ir in 0..nr {
        let t0 = pi * 0.5.as_() * (nr - 1 - ir).as_() / nr.as_();
        let y0 = -l * 0.5.as_() - r * t0.sin();
        let c0 = r * t0.cos();
        for ic in 0..nc {
            let theta = 2.as_() * pi * ic.as_() / nc.as_();
            vtx2xyz[(1 + ir * nc + ic) * 3 + 0] = c0 * theta.cos();
            vtx2xyz[(1 + ir * nc + ic) * 3 + 1] = y0;
            vtx2xyz[(1 + ir * nc + ic) * 3 + 2] = c0 * theta.sin();
        }
    }
    for il in 0..nl - 1 {
        let y0 = -l * 0.5.as_() + (il + 1).as_() * l / nl.as_();
        for ic in 0..nc {
            let theta = 2.as_() * pi * ic.as_() / nc.as_();
            vtx2xyz[(1 + (il + nr) * nc + ic) * 3 + 0] = r * theta.cos();
            vtx2xyz[(1 + (il + nr) * nc + ic) * 3 + 1] = y0;
            vtx2xyz[(1 + (il + nr) * nc + ic) * 3 + 2] = r * theta.sin();
        }
    }
    for ir in 0..nr {
        let t0 = pi * 0.5.as_() * ir.as_() / nr.as_();
        let y0 = l * 0.5.as_() + r * (t0).sin();
        let c0 = r * t0.cos();
        for ic in 0..nc {
            let theta = 2.as_() * pi * ic.as_() / nc.as_();
            vtx2xyz[(1 + (ir + nl + nr - 1) * nc + ic) * 3 + 0] = c0 * theta.cos();
            vtx2xyz[(1 + (ir + nl + nr - 1) * nc + ic) * 3 + 1] = y0;
            vtx2xyz[(1 + (ir + nl + nr - 1) * nc + ic) * 3 + 2] = c0 * theta.sin();
        }
    }
    {
        let np = vtx2xyz.len() / 3;
        vtx2xyz[(np - 1) * 3 + 0] = 0.as_();
        vtx2xyz[(np - 1) * 3 + 1] = l * 0.5.as_() + r;
        vtx2xyz[(np - 1) * 3 + 2] = 0.as_();
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_capsule_tri3() {
    capsule_tri3::<f32>(1., 1., 32, 12, 5);
    capsule_tri3::<f64>(1., 1., 32, 12, 5);
}


pub fn capsule_tri3_connecting_two_point<T>(
    p0: [T;3],
    p1: [T;3],
    rad: T,
    nc: usize,
    nr: usize,
    nl: usize) -> (Vec<usize>, Vec<T>)
where T: nalgebra::RealField + Copy + num_traits::Float,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>
{
    let p0 = nalgebra::Vector3::<T>::from_column_slice(&p0);
    let p1 = nalgebra::Vector3::<T>::from_column_slice(&p1);
    let len = (p1-p0).norm();
    let (tri2vtx, mut vtx2xyz) = capsule_tri3(rad, len, nc, nr, nl);
    let q2 = nalgebra::Vector3::<T>::new(T::zero(), len*0.5_f64.as_(),T::zero());
    let mat = del_geo::mat3::minimum_rotation_matrix(
        nalgebra::Vector3::<T>::new(T::zero(), T::one(), T::zero()),
        (p1-p0).normalize());
    for v in vtx2xyz.chunks_mut(3) {
        let q0 =  nalgebra::Vector3::<T>::new(v[0],v[1],v[2]);
        let q1 = mat * (q0+q2) + p0;
        v[0] = q1.x;
        v[1] = q1.y;
        v[2] = q1.z;
    }
    (tri2vtx, vtx2xyz)
}


// --------------------------------------------------------


pub fn grid_quad2<T>(
    nx: usize,
    ny: usize) -> (Vec<usize>, Vec<T>)
    where T: num_traits::Float + 'static,
          f32: AsPrimitive<T>,
          usize: AsPrimitive<T>
{
    let np = (nx + 1) * (ny + 1);
    let mut vtx2xy: Vec<T> = vec![0_f32.as_(); np * 2];
    for iy in 0..ny + 1 {
        for ix in 0..nx + 1 {
            let ip = iy * (nx + 1) + ix;
            vtx2xy[ip * 2 + 0] = ix.as_();
            vtx2xy[ip * 2 + 1] = iy.as_();
        }
    }
    let mut quad2vtx = vec![0; nx * ny * 4];
    for iy in 0..ny {
        for ix in 0..nx {
            let iq = iy * nx + ix;
            quad2vtx[iq * 4 + 0] = (iy + 0) * (nx + 1) + (ix + 0);
            quad2vtx[iq * 4 + 1] = (iy + 0) * (nx + 1) + (ix + 1);
            quad2vtx[iq * 4 + 2] = (iy + 1) * (nx + 1) + (ix + 1);
            quad2vtx[iq * 4 + 3] = (iy + 1) * (nx + 1) + (ix + 0);
        }
    }
    (quad2vtx, vtx2xy)
}

#[test]
fn test_grid_quad2() {
    grid_quad2::<f32>(12, 5);
    grid_quad2::<f64>(12, 5);
}


pub fn torus_tri3<T>(
    radius_: T, // latitude
    radius_tube_: T, // meridian
    nlg: usize, // latitude
    nlt: usize) // meridian
    -> (Vec<usize>, Vec<T>)
    where T: num_traits::Float + 'static,
          f32: AsPrimitive<T>,
          usize: AsPrimitive<T>
{
    let rlg: T = (std::f32::consts::PI * 2_f32).as_() / nlg.as_();  // latitude
    let rlt: T = (std::f32::consts::PI * 2_f32).as_() / nlt.as_();
    let mut vtx2xyz: Vec<T> = vec![0_f32.as_(); nlg * nlt * 3];
    for ilg in 0..nlg {
        for ilt in 0..nlt {
            let r0: T = radius_ + radius_tube_ * (ilt.as_() * rlt).cos();
            vtx2xyz[(ilg * nlt + ilt) * 3 + 0] = r0 * (ilg.as_() * rlg).sin();
            vtx2xyz[(ilg * nlt + ilt) * 3 + 1] = r0 * (ilg.as_() * rlg).cos();
            vtx2xyz[(ilg * nlt + ilt) * 3 + 2] = radius_tube_ * (ilt.as_() * rlt).sin();
        }
    }
    let mut tri2vtx = vec![0; nlg * nlt * 2 * 3];
    for ilg in 0..nlg {
        for ilt in 0..nlt {
            let iug = if ilg == nlg - 1 { 0 } else { ilg + 1 };
            let iut = if ilt == nlt - 1 { 0 } else { ilt + 1 };
            tri2vtx[(ilg * nlt + ilt) * 6 + 0] = ilg * nlt + ilt;
            tri2vtx[(ilg * nlt + ilt) * 6 + 2] = iug * nlt + ilt;
            tri2vtx[(ilg * nlt + ilt) * 6 + 1] = iug * nlt + iut;
            tri2vtx[(ilg * nlt + ilt) * 6 + 3] = ilg * nlt + ilt;
            tri2vtx[(ilg * nlt + ilt) * 6 + 5] = iug * nlt + iut;
            tri2vtx[(ilg * nlt + ilt) * 6 + 4] = ilg * nlt + iut;
        }
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_torus_tri3() {
    torus_tri3::<f64>(1., 1., 32, 32);
    torus_tri3::<f32>(1., 1., 32, 32);
}

// --------------

pub fn sphere_tri3<T>(
    radius: T,
    n_longitude: usize,
    n_latitude: usize)  -> (Vec<usize>, Vec<T>)
    where T: num_traits::Float + 'static,
          f32: AsPrimitive<T>,
          usize: AsPrimitive<T>
{
    let mut vtx2xyz = Vec::<T>::new();
    let mut tri2vtx = Vec::<usize>::new();
    vtx2xyz.clear();
    if n_longitude <= 1 || n_latitude <= 2 {
        return (tri2vtx, vtx2xyz);
    }
    let pi: T = 3.1415926535_f32.as_();
    let dl: T = pi / n_longitude.as_();
    let dr: T = 2.as_() * pi / n_latitude.as_();
    vtx2xyz.reserve((n_latitude * (n_longitude - 1) + 2) * 3);
    for ila in 0..n_longitude + 1 {
        let y0 = (dl * ila.as_()).cos();
        let r0 = (dl * ila.as_()).sin();
        for ilo in 0..n_latitude {
            let x0 = r0 * (dr * ilo.as_()).sin();
            let z0 = r0 * (dr * ilo.as_()).cos();
            vtx2xyz.push(radius * x0);
            vtx2xyz.push(radius * y0);
            vtx2xyz.push(radius * z0);
            if ila == 0 || ila == n_longitude { break; }
        }
    }
//
    let ntri = n_latitude * (n_longitude - 1) * 2 + n_latitude * 2;
    tri2vtx.reserve(ntri * 3);
    for ilo in 0..n_latitude {
        tri2vtx.push(0);
        tri2vtx.push((ilo + 0) % n_latitude + 1);
        tri2vtx.push((ilo + 1) % n_latitude + 1);
    }
    for ila in 0..n_longitude -2 {
        for ilo in 0..n_latitude {
            let i1 = (ila + 0) * n_latitude + 1 + (ilo + 0) % n_latitude;
            let i2 = (ila + 0) * n_latitude + 1 + (ilo + 1) % n_latitude;
            let i3 = (ila + 1) * n_latitude + 1 + (ilo + 1) % n_latitude;
            let i4 = (ila + 1) * n_latitude + 1 + (ilo + 0) % n_latitude;
            tri2vtx.push(i3);
            tri2vtx.push(i2);
            tri2vtx.push(i1);
            tri2vtx.push(i4);
            tri2vtx.push(i3);
            tri2vtx.push(i1);
        }
    }
    for ilo in 0..n_latitude {
        tri2vtx.push(n_latitude * (n_longitude - 1) + 1);
        tri2vtx.push((n_longitude - 2) * n_latitude + 1 + (ilo + 1) % n_latitude);
        tri2vtx.push((n_longitude - 2) * n_latitude + 1 + (ilo + 0) % n_latitude);
    }
    (tri2vtx, vtx2xyz)
}