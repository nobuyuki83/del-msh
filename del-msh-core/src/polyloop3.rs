//! methods for 3D poly loop

use num_traits::AsPrimitive;

pub fn vtx2framex<T>(vtx2xyz: &[T]) -> Vec<T>
where
    T: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2bin = vec![T::zero(); num_vtx * 3];
    {
        // first segment
        let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, 0);
        let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, 1);
        let v01 = p1.sub(p0);
        let (x, _) = del_geo_core::vec3::basis_xy_from_basis_z(&v01);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, 0).copy_from_slice(&x);
    }
    for iseg1 in 1..num_vtx {
        // parallel transport
        let iv0 = iseg1 - 1;
        let iv1 = iseg1;
        let iv2 = (iseg1 + 1) % num_vtx;
        let iseg0 = iseg1 - 1;
        let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, iv0);
        let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, iv1);
        let p2 = crate::vtx2xyz::to_vec3(vtx2xyz, iv2);
        let v01 = p1.sub(p0);
        let v12 = p2.sub(p1);
        let rot = del_geo_core::mat3_col_major::minimum_rotation_matrix(&v01, &v12);
        let b01 = crate::vtx2xyz::to_vec3(&vtx2bin, iseg0);
        let b12 = del_geo_core::mat3_col_major::mult_vec(&rot, b01);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, iseg1).copy_from_slice(&b12);
    }
    vtx2bin
}

pub fn framez<T>(vtx2xyz: &[T], i_vtx: usize) -> [T; 3]
where
    T: num_traits::Float + Copy,
{
    let num_vtx = vtx2xyz.len() / 3;
    assert!(i_vtx < num_vtx);
    let i0_vtx = (i_vtx + num_vtx - 1) % num_vtx;
    // let i1_vtx = i_vtx;
    let i2_vtx = (i_vtx + 1) % num_vtx;
    let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, i0_vtx);
    let p2 = crate::vtx2xyz::to_vec3(vtx2xyz, i2_vtx);
    use del_geo_core::vec3::Vec3;
    p2.sub(p0).normalize()
}

fn match_frames_of_two_ends<T>(vtx2xyz: &[T], vtx2bin0: &[T]) -> Vec<T>
where
    T: num_traits::Float + Copy + 'static + std::fmt::Display,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    let theta = {
        let x0 = crate::vtx2xyz::to_vec3(vtx2bin0, 0);
        let p0 = &crate::vtx2xyz::to_vec3(vtx2xyz, 0);
        let p1 = &crate::vtx2xyz::to_vec3(vtx2xyz, 1);
        let v01 = p1.sub(p0).normalize();
        assert!(x0.dot(&v01).abs() < 1.0e-6_f64.as_());
        let xn = crate::vtx2xyz::to_vec3(vtx2bin0, num_vtx - 1);
        let pn = &crate::vtx2xyz::to_vec3(vtx2xyz, num_vtx - 1);
        let vn0 = p0.sub(pn).normalize();
        let rot = del_geo_core::mat3_col_major::minimum_rotation_matrix(&vn0, &v01);
        let x1a = del_geo_core::mat3_col_major::mult_vec(&rot, xn);
        let y0 = v01.cross(x0);
        assert!(
            x1a.dot(&v01).abs() < 1.0e-4f64.as_(),
            "{}",
            x1a.dot(&v01).abs()
        );
        assert!((y0.norm() - 1.0_f64.as_()).abs() < 1.0e-6_f64.as_());
        let c0 = x1a.dot(x0);
        let s0 = x1a.dot(&y0);
        T::atan2(s0, c0)
    };
    let theta_step = theta / num_vtx.as_();
    let mut vtx2bin1 = vec![T::zero(); num_vtx * 3];
    for iseg in 0..num_vtx {
        let dtheta = theta_step * iseg.as_();
        let x0 = crate::vtx2xyz::to_vec3(vtx2bin0, iseg);
        let ivtx0 = iseg;
        let ivtx1 = (iseg + 1) % num_vtx;
        let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx1);
        let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx0);
        let v01 = p1.sub(p0).normalize();
        let y0 = v01.cross(x0);
        assert!(
            (x0.cross(&y0).dot(&v01) - 1.as_()).abs() < 1.0e-3_f64.as_(),
            "{}",
            x0.cross(&y0).dot(&v01)
        );
        let x0 = x0.scale(dtheta.sin());
        let y0 = y0.scale(dtheta.cos());
        let x1 = x0.add(&y0);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin1, iseg).copy_from_slice(&x1);
    }
    vtx2bin1
}

pub fn smooth_frame<T>(vtx2xyz: &[T]) -> Vec<T>
where
    T: num_traits::Float + 'static + Copy + std::fmt::Display,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    let vtx2bin0 = vtx2framex(vtx2xyz);
    // dbg!(&vtx2bin0);
    match_frames_of_two_ends(vtx2xyz, &vtx2bin0)
}

pub fn normal_binormal<T>(vtx2xyz: &[T]) -> (Vec<T>, Vec<T>)
where
    T: num_traits::Float + Copy,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2bin = vec![T::zero(); num_vtx * 3];
    let mut vtx2nrm = vec![T::zero(); num_vtx * 3];
    for ivtx1 in 0..num_vtx {
        let ivtx0 = (ivtx1 + num_vtx - 1) % num_vtx;
        let ivtx2 = (ivtx1 + 1) % num_vtx;
        let v0 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx0);
        let v1 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx1);
        let v2 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx2);
        let v01 = v1.sub(v0);
        let v12 = v2.sub(v1);
        let binormal = v12.cross(&v01).normalize();
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, ivtx1).copy_from_slice(&binormal);
        let norm = v01.add(&v12).cross(&binormal).normalize();
        crate::vtx2xyz::to_vec3_mut(&mut vtx2nrm, ivtx1).copy_from_slice(&norm);
    }
    (vtx2nrm, vtx2bin)
}

pub fn smooth_gradient_of_distance(vtx2xyz: &[f64], q: &[f64; 3]) -> [f64; 3] {
    use del_geo_core::vec3::Vec3;
    let n = vtx2xyz.len() / 3;
    let mut dd = [0f64; 3];
    for i_seg in 0..n {
        let ip0 = i_seg;
        let ip1 = (i_seg + 1) % n;
        let (_, dd0) = del_geo_core::edge3::wdw_integral_of_inverse_distance_cubic(
            q,
            crate::vtx2xyz::to_vec3(vtx2xyz, ip0),
            crate::vtx2xyz::to_vec3(vtx2xyz, ip1),
        );
        dd.add_in_place(&dd0);
    }
    dd
}

pub fn extend_avoid_intersection(
    p0: &[f64; 3],
    v0: &[f64; 3],
    vtx2xyz: &[f64],
    eps: f64,
    n: usize,
) -> [f64; 3] {
    use del_geo_core::vec3::Vec3;
    let mut p1 = p0.add(&v0.scale(eps));
    for _i in 0..n {
        let v1 = smooth_gradient_of_distance(vtx2xyz, &p1)
            .normalize()
            .scale(-1f64);
        p1.add_in_place(&v1.scale(eps));
    }
    p1
}

pub fn tube_mesh_avoid_intersection(
    vtx2xyz: &[f64],
    vtx2bin: &[f64],
    eps: f64,
    niter: usize,
) -> (Vec<usize>, Vec<f64>) {
    use del_geo_core::vec3::Vec3;
    let n = 8;
    let dtheta = std::f64::consts::PI * 2. / n as f64;
    let num_vtx = vtx2xyz.len() / 3;
    let mut pnt2xyz = Vec::<f64>::new();
    for ipnt in 0..num_vtx {
        let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, ipnt);
        let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, (ipnt + 1) % num_vtx);
        let z0 = p1.sub(p0).normalize();
        let x0 = crate::vtx2xyz::to_vec3(vtx2bin, ipnt);
        let y0 = z0.cross(x0);
        for i in 0..n {
            let theta = dtheta * i as f64;
            let x0 = x0.scale(theta.cos());
            let y0 = y0.scale(theta.sin());
            let v0 = x0.add(&y0);
            let q0 = extend_avoid_intersection(p0, &v0, vtx2xyz, eps, niter);
            // let q0 = p0 + v0.scale(rad);
            q0.iter().for_each(|&v| pnt2xyz.push(v));
        }
    }

    let mut tri2pnt = Vec::<usize>::new();
    for iseg in 0..num_vtx {
        let ipnt0 = iseg;
        let ipnt1 = (ipnt0 + 1) % num_vtx;
        for i in 0..n {
            tri2pnt.push(ipnt0 * n + i);
            tri2pnt.push(ipnt0 * n + (i + 1) % n);
            tri2pnt.push(ipnt1 * n + i);
            //
            tri2pnt.push(ipnt1 * n + (i + 1) % n);
            tri2pnt.push(ipnt1 * n + i);
            tri2pnt.push(ipnt0 * n + (i + 1) % n);
        }
    }
    (tri2pnt, pnt2xyz)
}

pub fn write_wavefrontobj<P: AsRef<std::path::Path>>(filepath: P, vtx2xyz: &[f32]) {
    use std::io::Write;
    let mut file = std::fs::File::create(filepath).expect("file not found.");
    for vtx in vtx2xyz.chunks(3) {
        writeln!(file, "v {} {} {}", vtx[0], vtx[1], vtx[2]).expect("fail");
    }
    write!(file, "l ").expect("fail");
    for i in 1..vtx2xyz.len() / 3 + 1 {
        write!(file, "{} ", i).expect("fail");
    }
    writeln!(file, "1").expect("fail");
}

pub fn nearest_to_edge3<T>(vtx2xyz: &[T], p0: &[T; 3], p1: &[T; 3]) -> (T, T, T)
where
    T: num_traits::Float + Copy + 'static,
    usize: AsPrimitive<T>,
{
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vtx2xyz.len(), num_vtx * 3);
    let mut res = (T::max_value(), T::zero(), T::zero());
    for i_edge in 0..num_vtx {
        let iv0 = i_edge;
        let iv1 = (i_edge + 1) % num_vtx;
        let q0 = crate::vtx2xyz::to_vec3(vtx2xyz, iv0);
        let q1 = crate::vtx2xyz::to_vec3(vtx2xyz, iv1);
        let (dist, r0, r1) = del_geo_core::edge3::nearest_to_edge3(p0, p1, q0, q1);
        if dist > res.0 {
            continue;
        }
        //dbg!((p0+(p1-p0)*r0));
        //dbg!((q0+(q1-q0)*r1));
        res.0 = dist;
        res.1 = <usize as AsPrimitive<T>>::as_(i_edge) + r1;
        res.2 = r0;
    }
    res
}

pub fn nearest_to_point3<T>(vtx2xyz: &[T], p0: &[T; 3]) -> (T, T)
where
    T: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    assert_eq!(p0.len(), 3);
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vtx2xyz.len(), num_vtx * 3);
    let mut res = (T::max_value(), T::zero());
    for i_edge in 0..num_vtx {
        let iv0 = i_edge;
        let iv1 = (i_edge + 1) % num_vtx;
        let q0 = crate::vtx2xyz::to_vec3(vtx2xyz, iv0);
        let q1 = crate::vtx2xyz::to_vec3(vtx2xyz, iv1);
        let (dist, rq) = del_geo_core::edge3::nearest_to_point3(q0, q1, p0);
        if dist < res.0 {
            //dbg!((p0+(p1-p0)*r0));
            //dbg!((q0+(q1-q0)*r1));
            res.0 = dist;
            res.1 = <usize as AsPrimitive<T>>::as_(i_edge) + rq;
        }
    }
    res
}

pub fn winding_number(vtx2xyz: &[f64], org: &[f64; 3], dir: &[f64; 3]) -> f64 {
    use del_geo_core::vec3::Vec3;
    use num_traits::FloatConst;
    //let org = nalgebra::Vector3::<f64>::from_row_slice(org);
    //let dir = nalgebra::Vector3::<f64>::from_row_slice(dir);
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vtx2xyz.len(), num_vtx * 3);
    let mut sum = 0.;
    for i_edge in 0..num_vtx {
        let iv0 = i_edge;
        let iv1 = (i_edge + 1) % num_vtx;
        let q0 = crate::vtx2xyz::to_vec3(vtx2xyz, iv0).sub(org);
        let q1 = crate::vtx2xyz::to_vec3(vtx2xyz, iv1).sub(org);
        let q0 = q0.sub(&dir.scale(q0.dot(dir)));
        let q1 = q1.sub(&dir.scale(q1.dot(dir)));
        let q0 = q0.normalize();
        let q1 = q1.normalize();
        let s = q0.cross(&q1).dot(dir);
        let c = q0.dot(&q1);
        sum += s.atan2(c);
    }
    sum * f64::FRAC_1_PI() * 0.5
}

#[allow(clippy::identity_op)]
pub fn position_from_barycentric_coordinate<T>(vtx2xyz: &[T], r: T) -> [T; 3]
where
    T: num_traits::Float + AsPrimitive<usize> + std::fmt::Display + std::fmt::Debug,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let ied: usize = r.as_();
    let ned = vtx2xyz.len() / 3;
    if r.as_() == ned {
        assert_eq!(ied.as_(), r);
        return *crate::vtx2xyz::to_vec3(vtx2xyz, 0);
    }
    assert!(ied < ned, "{}, {}, {}", r, ied, ned);
    let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, ied);
    let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, (ied + 1) % ned);
    let r0 = r - ied.as_();
    p0.add(&p1.sub(p0).scale(r0))
}

#[allow(clippy::identity_op)]
pub fn smooth<T>(vtx2xyz: &[T], r: T, num_iter: usize) -> Vec<T>
where
    T: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2xyz1 = Vec::from(vtx2xyz);
    for _iter in 0..num_iter {
        for ip1 in 0..num_vtx {
            let ip0 = (ip1 + num_vtx - 1) % num_vtx;
            let ip2 = (ip1 + 1) % num_vtx;
            let p0 = crate::vtx2xyz::to_vec3(&vtx2xyz1, ip0);
            let p1 = crate::vtx2xyz::to_vec3(&vtx2xyz1, ip1);
            let p2 = crate::vtx2xyz::to_vec3(&vtx2xyz1, ip2);
            let pm = p0.add(p2).scale(0.5f64.as_());
            let p1n = del_geo_core::edge3::position_from_ratio(p1, &pm, r);
            vtx2xyz1[ip1 * 3 + 0] = p1n[0];
            vtx2xyz1[ip1 * 3 + 1] = p1n[1];
            vtx2xyz1[ip1 * 3 + 2] = p1n[2];
        }
    }
    vtx2xyz1
}

/// TODO: it might be better to specify the normal vector
pub fn to_trimesh3_torus(
    vtx2xyz: &[f32],
    vtx2bin: &[f32],
    rad: f32,
    ndiv_circum: usize,
) -> (Vec<usize>, Vec<f32>) {
    use del_geo_core::vec3::Vec3;
    let n = ndiv_circum;
    let dtheta = std::f32::consts::PI * 2. / n as f32;
    let num_vtx = vtx2xyz.len() / 3;
    let mut pnt2xyz = Vec::<f32>::new();
    for ipnt in 0..num_vtx {
        let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, ipnt);
        let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, (ipnt + 1) % num_vtx);
        let z0 = p1.sub(p0).normalize();
        let x0 = crate::vtx2xyz::to_vec3(vtx2bin, ipnt);
        let y0 = z0.cross(x0);
        for i in 0..n {
            let theta = dtheta * i as f32;
            let v0 = x0.scale(theta.cos()).add(&y0.scale(theta.sin()));
            let q0 = p0.add(&v0.scale(rad));
            q0.iter().for_each(|&v| pnt2xyz.push(v));
        }
    }

    let mut tri2pnt = Vec::<usize>::new();
    for iseg in 0..num_vtx {
        let ipnt0 = iseg;
        let ipnt1 = (ipnt0 + 1) % num_vtx;
        for i in 0..n {
            tri2pnt.push(ipnt0 * n + i);
            tri2pnt.push(ipnt0 * n + (i + 1) % n);
            tri2pnt.push(ipnt1 * n + i);
            //
            tri2pnt.push(ipnt1 * n + (i + 1) % n);
            tri2pnt.push(ipnt1 * n + i);
            tri2pnt.push(ipnt0 * n + (i + 1) % n);
        }
    }
    (tri2pnt, pnt2xyz)
}
