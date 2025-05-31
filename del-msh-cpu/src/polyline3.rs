//! methods related to 3D polyline

use num_traits::AsPrimitive;

/// the center of gravity
pub fn cog<T>(vtx2xyz: &[T]) -> [T; 3]
where
    T: num_traits::Float,
{
    let one = T::one();
    let half = one / (one + one);
    let num_vtx = vtx2xyz.len() / 3;
    let mut cg = [T::zero(); 3];
    let mut w = T::zero();
    use del_geo_core::vec3::Vec3;
    for iseg in 0..num_vtx - 1 {
        let ip0 = iseg;
        let ip1 = iseg + 1;
        let p0 = arrayref::array_ref![vtx2xyz, ip0 * 3, 3];
        let p1 = arrayref::array_ref![vtx2xyz, ip1 * 3, 3];
        let len = p0.sub(p1).norm();
        cg = cg.add(&p0.add(p1).scale(half * len));
        w = w + len;
    }
    cg.scale(T::one() / w)
}

/// bi-normal vector on each vertex
pub fn vtx2framex<T>(vtx2xyz: &[T]) -> Vec<T>
where
    T: num_traits::Float,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2bin = vec![T::zero(); num_vtx * 3];
    {
        // first segment
        let p1 = arrayref::array_ref![vtx2xyz, 3, 3];
        let p0 = arrayref::array_ref![vtx2xyz, 0, 3];
        let v01 = p1.sub(p0);
        let (x, _) = del_geo_core::vec3::basis_xy_from_basis_z(&v01);
        vtx2bin[0..3].copy_from_slice(&x);
    }
    for i_seg1 in 1..num_vtx - 1 {
        let iv0 = i_seg1 - 1;
        let iv1 = i_seg1;
        let iv2 = i_seg1 + 1;
        let i_seg0 = i_seg1 - 1;
        let p0 = arrayref::array_ref![vtx2xyz, iv0 * 3, 3];
        let p1 = arrayref::array_ref![vtx2xyz, iv1 * 3, 3];
        let p2 = arrayref::array_ref![vtx2xyz, iv2 * 3, 3];
        let v01 = p1.sub(p0);
        let v12 = p2.sub(p1);
        let rot = del_geo_core::mat3_col_major::minimum_rotation_matrix(&v01, &v12);
        let b01: &[T; 3] = arrayref::array_ref![vtx2bin, i_seg0 * 3, 3];
        let b12: [T; 3] = del_geo_core::mat3_col_major::mult_vec(&rot, b01);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, i_seg1).copy_from_slice(&b12);
    }
    {
        let a: [T; 3] = crate::vtx2xyz::to_vec3(&vtx2bin, num_vtx - 2).to_owned();
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, num_vtx - 1).copy_from_slice(&a);
    }
    vtx2bin
}

#[test]
fn test_vtx2framex() {
    let num_vtx = 10usize;
    let length = 3.0f64;
    let vtx2xyz: Vec<f64> = (0..num_vtx)
        .flat_map(|i_vtx| [0., i_vtx as f64 / num_vtx as f64 * length, 0.])
        .collect();
    let vtx2framex = vtx2framex(&vtx2xyz);
    vtx2framex.chunks(3).for_each(|v| {
        let v = [v[0], v[1], v[2]];
        let len = del_geo_core::vec3::norm(&v);
        assert!((len - 1.0).abs() < 1.0e-8);
    });
}

pub fn framez<T>(vtx2xyz: &[T], i_vtx: usize) -> [T; 3]
where
    T: num_traits::Float + Copy,
{
    let num_vtx = vtx2xyz.len() / 3;
    assert!(i_vtx < num_vtx);
    use del_geo_core::vec3::Vec3;
    if i_vtx == 0 {
        let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, 0);
        let p2 = crate::vtx2xyz::to_vec3(vtx2xyz, 1);
        return p2.sub(p1).normalize();
    }
    if i_vtx == num_vtx - 1 {
        let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, num_vtx - 2);
        let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, num_vtx - 1);
        return p1.sub(p0).normalize();
    }
    let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, i_vtx - 1);
    let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, i_vtx);
    let p2 = crate::vtx2xyz::to_vec3(vtx2xyz, i_vtx + 1);
    let u01 = p1.sub(p0).normalize();
    let u12 = p2.sub(p1).normalize();
    u01.add(&u12).normalize()
}

pub fn vtx2framey<T>(vtx2xyz: &[T], vtx2framex: &[T]) -> Vec<T>
where
    T: num_traits::Float,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vtx2framex.len(), num_vtx * 3);
    let mut vtx2framey = vec![T::zero(); num_vtx * 3];
    for i_vtx in 0..num_vtx {
        let framez = framez(vtx2xyz, i_vtx);
        let framex = crate::vtx2xyz::to_vec3(vtx2framex, i_vtx);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2framey, i_vtx).copy_from_slice(&framez.cross(framex));
    }
    vtx2framey
}

pub fn normal_binormal<T>(vtx2xyz: &[T]) -> (Vec<T>, Vec<T>)
where
    T: num_traits::Float + Copy,
{
    use del_geo_core::vec3::Vec3;
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2bin = vec![T::zero(); num_vtx * 3];
    let mut vtx2nrm = vec![T::zero(); num_vtx * 3];
    for ivtx1 in 1..num_vtx - 1 {
        let ivtx0 = (ivtx1 + num_vtx - 1) % num_vtx;
        let ivtx2 = (ivtx1 + 1) % num_vtx;
        let v0 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx0);
        let v1 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx1);
        let v2 = crate::vtx2xyz::to_vec3(vtx2xyz, ivtx2);
        let v01 = v1.sub(v0);
        let v12 = v2.sub(v1);
        let binormal = v12.cross(&v01);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, ivtx1).copy_from_slice(&binormal.normalize());
        let norm = v01.add(&v12).cross(&binormal);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2nrm, ivtx1).copy_from_slice(&norm.normalize());
    }
    {
        let c1 = *crate::vtx2xyz::to_vec3(&vtx2nrm, 1);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2nrm, 0).copy_from_slice(&c1);
    }
    {
        let c1 = *crate::vtx2xyz::to_vec3(&vtx2nrm, num_vtx - 2);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2nrm, num_vtx - 1).copy_from_slice(&c1);
    }
    {
        let c1 = *crate::vtx2xyz::to_vec3(&vtx2bin, 1);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, 0).copy_from_slice(&c1);
    }
    {
        let c1 = *crate::vtx2xyz::to_vec3(&vtx2bin, num_vtx - 2);
        crate::vtx2xyz::to_vec3_mut(&mut vtx2bin, num_vtx - 1).copy_from_slice(&c1);
    }
    (vtx2nrm, vtx2bin)
}

pub fn set_vtx2xyz_for_generalized_cylinder_open_end<Index, T>(
    vtx2xyz: &mut [T],
    vtxl2xyz: &[T],
    rad: T,
) where
    T: num_traits::Float + num_traits::FloatConst + 'static,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let one = T::one();
    let two = one + one;
    let pi = T::PI();
    let num_vtxl = vtxl2xyz.len() / 3;
    let num_vtx = vtx2xyz.len() / 3;
    let ndiv_circum = num_vtx / num_vtxl;
    let vtxl2framex = vtx2framex(vtxl2xyz);
    let vtxl2framey = vtx2framey(vtxl2xyz, &vtxl2framex);
    for i_vtxl in 0..num_vtxl {
        let p0 = crate::vtx2xyz::to_vec3(vtxl2xyz, i_vtxl);
        let ex = crate::vtx2xyz::to_vec3(&vtxl2framex, i_vtxl);
        let ey = crate::vtx2xyz::to_vec3(&vtxl2framey, i_vtxl);
        for ic in 0..ndiv_circum {
            let theta = two * pi * ic.as_() / ndiv_circum.as_();
            let ay = ey.scale(rad * num_traits::Float::cos(theta));
            let ax = ex.scale(rad * num_traits::Float::sin(theta));
            let q = p0.add(&ay).add(&ax);
            vtx2xyz[(i_vtxl * ndiv_circum + ic) * 3] = q[0];
            vtx2xyz[(i_vtxl * ndiv_circum + ic) * 3 + 1] = q[1];
            vtx2xyz[(i_vtxl * ndiv_circum + ic) * 3 + 2] = q[2];
        }
    }
}

#[allow(clippy::identity_op)]
pub fn to_trimesh3_capsule<T>(
    vtxl2xyz: &[T],
    ndiv_circum: usize,
    ndiv_longtitude: usize,
    r: T,
) -> (Vec<usize>, Vec<T>)
where
    T: num_traits::Float + Copy + num_traits::FloatConst + 'static,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    assert!(ndiv_circum > 2);
    let num_vtxl = vtxl2xyz.len() / 3;
    let vtxl2framex = vtx2framex(vtxl2xyz);
    let vtxl2framey = vtx2framey(vtxl2xyz, &vtxl2framex);
    //
    let ndiv_length = vtxl2xyz.len() / 3 - 1;
    let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::cylinder_closed_end_yup::<T>(
        T::one(),
        T::one(),
        ndiv_circum,
        2 * ndiv_longtitude + ndiv_length - 2,
        true,
    );
    let tri2vtx = Vec::<usize>::from(tri2vtx.as_slice());
    let mut vtx2xyz = Vec::<T>::from(vtx2xyz.as_slice());
    assert_eq!(
        vtx2xyz.len() / 3,
        (2 * ndiv_longtitude + ndiv_length - 1) * ndiv_circum + 2
    );
    let pi: T = T::PI();
    let one = T::one();
    let half: T = one / (one + one);
    {
        // south pole
        let p0 = crate::vtx2xyz::to_vec3(vtxl2xyz, 0);
        let ez = framez(vtxl2xyz, 0);
        let q = p0.sub(&ez.scale(r));
        vtx2xyz[0] = q[0];
        vtx2xyz[1] = q[1];
        vtx2xyz[2] = q[2];
    }
    for ir in 0..ndiv_longtitude {
        let p0 = crate::vtx2xyz::to_vec3(vtxl2xyz, 0);
        let ex = crate::vtx2xyz::to_vec3(&vtxl2framex, 0);
        let ey = crate::vtx2xyz::to_vec3(&vtxl2framey, 0);
        let ez = framez(vtxl2xyz, 0);
        let t0 = pi * half * (ndiv_longtitude - 1 - ir).as_() / ndiv_longtitude.as_();
        let c0 = r * num_traits::Float::cos(t0);
        for ic in 0..ndiv_circum {
            let theta = 2.as_() * pi * ic.as_() / ndiv_circum.as_();
            let az = ez.scale(-r * num_traits::Float::sin(t0));
            let ay = ey.scale(c0 * num_traits::Float::cos(theta));
            let ax = ex.scale(c0 * num_traits::Float::sin(theta));
            let q = p0.add(&az).add(&ay).add(&ax);
            vtx2xyz[(1 + ir * ndiv_circum + ic) * 3 + 0] = q[0];
            vtx2xyz[(1 + ir * ndiv_circum + ic) * 3 + 1] = q[1];
            vtx2xyz[(1 + ir * ndiv_circum + ic) * 3 + 2] = q[2];
        }
    }
    for il in 0..ndiv_length - 1 {
        let p0 = crate::vtx2xyz::to_vec3(vtxl2xyz, il + 1);
        let ex = crate::vtx2xyz::to_vec3(&vtxl2framex, il + 1);
        let ey = crate::vtx2xyz::to_vec3(&vtxl2framey, il + 1);
        for ic in 0..ndiv_circum {
            let theta = 2.as_() * pi * ic.as_() / ndiv_circum.as_();
            let ay = ey.scale(r * num_traits::Float::cos(theta));
            let ax = ex.scale(r * num_traits::Float::sin(theta));
            let q = p0.add(&ay).add(&ax);
            vtx2xyz[(1 + (il + ndiv_longtitude) * ndiv_circum + ic) * 3 + 0] = q[0];
            vtx2xyz[(1 + (il + ndiv_longtitude) * ndiv_circum + ic) * 3 + 1] = q[1];
            vtx2xyz[(1 + (il + ndiv_longtitude) * ndiv_circum + ic) * 3 + 2] = q[2];
        }
    }
    for ir in 0..ndiv_longtitude {
        let p0 = crate::vtx2xyz::to_vec3(vtxl2xyz, num_vtxl - 1);
        let ex = crate::vtx2xyz::to_vec3(&vtxl2framex, num_vtxl - 1);
        let ey = crate::vtx2xyz::to_vec3(&vtxl2framey, num_vtxl - 1);
        let ez = framez(vtxl2xyz, num_vtxl - 1);
        let t0 = pi * half * ir.as_() / ndiv_longtitude.as_();
        let c0 = r * num_traits::Float::cos(t0);
        for ic in 0..ndiv_circum {
            let theta = 2.as_() * pi * ic.as_() / ndiv_circum.as_();
            let az = ez.scale(r * num_traits::Float::sin(t0));
            let ay = ey.scale(c0 * num_traits::Float::cos(theta));
            let ax = ex.scale(c0 * num_traits::Float::sin(theta));
            let q = p0.add(&az).add(&ax).add(&ay);
            vtx2xyz[(1 + (ir + ndiv_length + ndiv_longtitude - 1) * ndiv_circum + ic) * 3 + 0] =
                q[0];
            vtx2xyz[(1 + (ir + ndiv_length + ndiv_longtitude - 1) * ndiv_circum + ic) * 3 + 1] =
                q[1];
            vtx2xyz[(1 + (ir + ndiv_length + ndiv_longtitude - 1) * ndiv_circum + ic) * 3 + 2] =
                q[2];
        }
    }
    {
        // North Pole
        let p0 = crate::vtx2xyz::to_vec3(vtxl2xyz, num_vtxl - 1);
        let ez = framez(vtxl2xyz, num_vtxl - 1);
        let q = p0.add(&ez.scale(r));
        let np = vtx2xyz.len() / 3;
        vtx2xyz[(np - 1) * 3 + 0] = q[0];
        vtx2xyz[(np - 1) * 3 + 1] = q[1];
        vtx2xyz[(np - 1) * 3 + 2] = q[2];
    }
    (tri2vtx, vtx2xyz)
}

/*
pub fn nearest_to_polyline3<T>(
    poly_a: &Vec::<nalgebra::Vector3::<T>>,
    poly_b: &Vec::<nalgebra::Vector3::<T>>) -> (T,T,T)
    where T: nalgebra::RealField + Copy,
          f64: AsPrimitive<T>,
          usize: AsPrimitive<T>
{
    let mut res: (T,T,T) = (T::max_value().unwrap(), T::zero(), T::zero());
    for ia in 0..poly_a.len() -1 {
        let a0 = &poly_a[ia];
        let a1 = &poly_a[ia+1];
        for ib in 0..poly_b.len() -1 {
            let b0 = &poly_b[ib];
            let b1 = &poly_b[ib+1];
            let dis = del_geo::edge3::nearest_to_edge3(a0,a1,b0,b1);
            if dis.0 < res.0 {
                res.0 = dis.0;
                res.1 = ia.as_() + dis.1;
                res.2 = ib.as_() + dis.2;
            }
        }
    }
    res
}
*/

pub fn contacting_pair(poly2vtx: &[usize], vtx2xyz: &[f32], dist0: f32) -> (Vec<usize>, Vec<f32>) {
    let num_poly = poly2vtx.len() - 1;
    let mut pair_idx = Vec::<usize>::new();
    let mut pair_prm = Vec::<f32>::new();
    for i_poly in 0..num_poly {
        for j_poly in i_poly + 1..num_poly {
            for i_seg in poly2vtx[i_poly]..poly2vtx[i_poly + 1] - 1 {
                let pi = crate::vtx2xyz::to_vec3(vtx2xyz, i_seg);
                let qi = crate::vtx2xyz::to_vec3(vtx2xyz, i_seg + 1);
                for j_seg in poly2vtx[j_poly]..poly2vtx[j_poly + 1] - 1 {
                    let pj = crate::vtx2xyz::to_vec3(vtx2xyz, j_seg);
                    let qj = crate::vtx2xyz::to_vec3(vtx2xyz, j_seg + 1);
                    let (dist, ri, rj) = del_geo_core::edge3::nearest_to_edge3(pi, qi, pj, qj);
                    if dist > dist0 {
                        continue;
                    }
                    pair_idx.extend([i_poly, j_poly]);
                    pair_prm.push((i_seg - poly2vtx[i_poly]) as f32 + ri);
                    pair_prm.push((j_seg - poly2vtx[j_poly]) as f32 + rj);
                }
            }
        }
    }
    (pair_idx, pair_prm)
}

pub fn position_from_barycentric_coordinate<T>(vtx2xyz: &[T], r: T) -> [T; 3]
where
    T: num_traits::Float + AsPrimitive<usize>,
    usize: AsPrimitive<T>,
{
    let ied: usize = r.as_();
    let ned = vtx2xyz.len() / 3 - 1;
    // dbg!(r, ied, ned);
    assert!(ied < ned);
    let p0 = crate::vtx2xyz::to_vec3(vtx2xyz, ied);
    let p1 = crate::vtx2xyz::to_vec3(vtx2xyz, ied + 1);
    let r0 = r - ied.as_();
    use del_geo_core::vec3::Vec3;
    p0.add(&p1.sub(p0).scale(r0))
}

pub fn smooth<T>(vtx2xyz: &[T], r: T, num_iter: usize) -> Vec<T>
where
    T: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let one = T::one();
    let half = one / (one + one);
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2xyz1 = Vec::from(vtx2xyz);
    for _iter in 0..num_iter {
        for ip1 in 1..num_vtx - 1 {
            let ip0 = (ip1 + num_vtx - 1) % num_vtx;
            let ip2 = (ip1 + 1) % num_vtx;
            let p0 = crate::vtx2xyz::to_vec3(&vtx2xyz1, ip0);
            let p1 = crate::vtx2xyz::to_vec3(&vtx2xyz1, ip1);
            let p2 = crate::vtx2xyz::to_vec3(&vtx2xyz1, ip2);
            let pm = p0.add(p2).scale(half);
            let p1n = pm.scale(r).add(&p1.scale(one - r));
            vtx2xyz1[ip1 * 3] = p1n[0];
            vtx2xyz1[ip1 * 3 + 1] = p1n[1];
            vtx2xyz1[ip1 * 3 + 2] = p1n[2];
        }
    }
    vtx2xyz1
}

pub fn length<T>(vtx2xyz: &[T]) -> T
where
    T: num_traits::Float + std::ops::AddAssign + std::fmt::Debug,
{
    assert_eq!(vtx2xyz.len() % 3, 0);
    let num_vtx = vtx2xyz.len() / 3;
    let mut len = T::zero();
    for i_vtx in 0..num_vtx - 1 {
        let j_vtx = i_vtx + 1;
        let elen = del_geo_core::edge3::length(
            arrayref::array_ref!(vtx2xyz, i_vtx * 3, 3),
            arrayref::array_ref!(vtx2xyz, j_vtx * 3, 3),
        );
        len += elen;
    }
    len
}
fn reduce_recursive(
    vtx2flg: &mut [i32],
    i_vtx0: usize,
    i_vtx1: usize,
    vtx2xyz: &[f32],
    threshold: f32,
) {
    use del_geo_core::vec3::Vec3;
    assert!(i_vtx1 as i64 - i_vtx0 as i64 > 1);
    let p0 = arrayref::array_ref![vtx2xyz, i_vtx0 * 3, 3];
    let p1 = arrayref::array_ref![vtx2xyz, i_vtx1 * 3, 3];
    let vtx_farest = {
        let mut vtx_nearest = (usize::MAX, 0.);
        for i_vtxm in i_vtx0 + 1..i_vtx1 {
            let pm = arrayref::array_ref![vtx2xyz, i_vtxm * 3, 3];
            let (_dist, rn) = del_geo_core::edge3::nearest_to_point3(p0, p1, pm);
            let pn = del_geo_core::vec3::axpy(rn, &p1.sub(p0), p0);
            let dist = del_geo_core::edge3::length(pm, &pn);
            if dist >= vtx_nearest.1 {
                vtx_nearest = (i_vtxm, dist);
            }
        }
        vtx_nearest
    };
    assert_ne!(vtx_farest.0, usize::MAX);
    assert!(i_vtx0 < vtx_farest.0);
    assert!(vtx_farest.0 < i_vtx1);
    if vtx_farest.1 > threshold {
        vtx2flg[vtx_farest.0] = 1; // this is fixed
    }
    if vtx_farest.0 - i_vtx0 > 1 {
        reduce_recursive(vtx2flg, i_vtx0, vtx_farest.0, vtx2xyz, threshold);
    }
    if i_vtx1 - vtx_farest.0 > 1 {
        reduce_recursive(vtx2flg, vtx_farest.0, i_vtx1, vtx2xyz, threshold);
    }
}

/// <https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm>
pub fn reduce(vtx2xyz: &[f32], threshold: f32) -> Vec<f32> {
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2flg = vec![0; num_vtx]; // 0: free, 1: fix
    vtx2flg[0] = 1;
    vtx2flg[num_vtx - 1] = 1;
    reduce_recursive(&mut vtx2flg, 0, num_vtx - 1, vtx2xyz, threshold);
    // dbg!(&vtx2flg);
    let vtx2xyz_reduced: Vec<f32> = vtx2xyz
        .chunks(3)
        .enumerate()
        .filter(|&(iv, _xyz)| vtx2flg[iv] == 1)
        .flat_map(|(_iv, xyz)| [xyz[0], xyz[1], xyz[2]])
        .collect();
    vtx2xyz_reduced
}

#[test]
fn test_reduce() -> anyhow::Result<()> {
    let vtx2xy = crate::polyloop2::from_circle(1.0, 100);
    let vtx2xyz = crate::vtx2xy::to_vtx2xyz(&vtx2xy);
    let vtx2xyz_reduced = reduce(&vtx2xyz, 0.01);
    crate::io_obj::save_vtx2xyz_as_polyloop("../target/reduce_polyline.obj", &vtx2xyz_reduced, 3)?;
    Ok(())
}
