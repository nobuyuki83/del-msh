//! methods that generate meshes of primitive shapes (e.g., cylinder, torus)

use crate::trimesh3::TriMesh3Owned;
use num_traits::AsPrimitive;

/// this function is for external allocation of memory
pub fn cylinder_open_end_number_of_tri_and_vtx(
    ndiv_circumference: usize,
    ndiv_side: usize,
) -> (usize, usize) {
    let num_vtx = ndiv_circumference * (ndiv_side + 1);
    let num_tri = ndiv_side * ndiv_circumference * 2;
    (num_tri, num_vtx)
}

pub fn cylinder_open_end_set_topology<Index>(tri2vtx: &mut [[Index; 3]], num_vtxc: usize)
where
    Index: num_traits::PrimInt + 'static,
    usize: AsPrimitive<Index>,
{
    let ndiv_side = num_vtxc - 1;
    let ndiv_circumference = tri2vtx.len() / (2 * ndiv_side);
    assert_eq!(tri2vtx.len(), ndiv_side * ndiv_circumference * 2);
    for i_side in 0..ndiv_side {
        for i_edge in 0..ndiv_circumference {
            let i0_vtx = i_side * ndiv_circumference + i_edge;
            let i1_vtx = i_side * ndiv_circumference + (i_edge + 1) % ndiv_circumference;
            let i2_vtx = i0_vtx + ndiv_circumference;
            let i3_vtx = i1_vtx + ndiv_circumference;
            let i_quad = i_side * ndiv_circumference + i_edge;
            let (i0_vtx, i1_vtx, i2_vtx, i3_vtx) =
                (i0_vtx.as_(), i1_vtx.as_(), i2_vtx.as_(), i3_vtx.as_());
            tri2vtx[i_quad * 2][0] = i0_vtx;
            tri2vtx[i_quad * 2][1] = i3_vtx;
            tri2vtx[i_quad * 2][2] = i1_vtx;
            tri2vtx[i_quad * 2 + 1][0] = i0_vtx;
            tri2vtx[i_quad * 2 + 1][1] = i2_vtx;
            tri2vtx[i_quad * 2 + 1][2] = i3_vtx;
        }
    }
}

/// generate 3D mesh of open cylinder
/// * `radius` - radius
/// * 'length' - length
pub fn cylinder_open_end_yup<T>(
    ndiv_circumference: usize,
    ndiv_side: usize,
    radius: T,
    length: T,
    is_center: bool,
) -> (Vec<[usize; 3]>, Vec<[T; 3]>)
where
    T: num_traits::Float + num_traits::FloatConst + Copy + 'static,
    usize: AsPrimitive<T>,
{
    assert!(ndiv_circumference > 2);
    let (num_tri, num_vtx) = cylinder_open_end_number_of_tri_and_vtx(ndiv_circumference, ndiv_side);
    //
    let mut vtx2xyz = vec![[T::zero(); 3]; num_vtx * 3];
    let two = T::one() + T::one();
    let half = T::one() / two;
    let pi: T = T::PI();
    let dr: T = two * pi / ndiv_circumference.as_();
    let y_min = if is_center { -length * half } else { T::zero() };
    for is in 0..ndiv_side + 1 {
        let y0: T = y_min + length * is.as_() / ndiv_side.as_();
        for ilo in 0..ndiv_circumference {
            let x0 = radius * (dr * ilo.as_()).cos();
            let z0 = radius * (dr * ilo.as_()).sin();
            let i_vtx = is * ndiv_circumference + ilo;
            //crate::vtx2xyz::to_vec3_mut(&mut vtx2xyz, i_vtx).copy_from_slice(&[x0, y0, z0]);
            vtx2xyz[i_vtx] = [x0, y0, z0];
        }
    }
    // ------------------------------------
    let mut tri2vtx = vec![[0usize; 3]; num_tri];
    cylinder_open_end_set_topology::<usize>(&mut tri2vtx, ndiv_side + 1);
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_cylinder_open_end_tri3() {
    cylinder_open_end_yup::<f32>(32, 16, 1f32, 1f32, true);
    cylinder_open_end_yup::<f64>(32, 16, 1f64, 1f64, true);
}

// -------------------------------------

pub fn cylinder_open_connecting_two_points<Real>(
    ndiv_circumference: usize,
    r: Real,
    p0: &[Real; 3],
    p1: &[Real; 3],
) -> (Vec<[usize; 3]>, Vec<[Real; 3]>)
where
    Real: 'static + num_traits::Float + Copy + num_traits::FloatConst,
    usize: AsPrimitive<Real>,
{
    use del_geo_core::vec3::Vec3;
    let len = p1.sub(p0).norm();
    let (tri2vtx, vtx2xyz0) = cylinder_open_end_yup(ndiv_circumference, 1, r, len, false);
    let rot = del_geo_core::mat3_col_major::minimum_rotation_matrix(
        &[Real::zero(), Real::one(), Real::zero()],
        &p1.sub(p0).normalize(),
    );
    let mut vtx2xyz1 = crate::vtx2xyz::transform_linear(&vtx2xyz0, &rot);
    vtx2xyz1.iter_mut().for_each(|xyz| {
        xyz[0] = xyz[0] + p0[0];
        xyz[1] = xyz[1] + p0[1];
        xyz[2] = xyz[2] + p0[2];
    });
    (tri2vtx, vtx2xyz1)
}

// -------------------------------------

#[allow(clippy::identity_op)]
fn cylinder_closed_end_topology<Index>(
    ndiv_side: usize,
    ndiv_circumference: usize,
) -> Vec<[Index; 3]>
where
    Index: num_traits::PrimInt + 'static + Copy,
    usize: AsPrimitive<Index>,
{
    let ndiv_longtitude = ndiv_side + 2;
    let num_tri = ndiv_circumference * (ndiv_longtitude - 1) * 2 + ndiv_circumference * 2;
    let mut tri2vtx = Vec::<[Index; 3]>::with_capacity(num_tri);
    for ic in 0..ndiv_circumference {
        tri2vtx.push([
            Index::zero(),
            (ic + 1).as_(),
            ((ic + 1) % ndiv_circumference + 1).as_(),
        ]);
    }
    for ih in 0..ndiv_longtitude - 2 {
        for ic in 0..ndiv_circumference {
            let i1 = (ih + 0) * ndiv_circumference + 1 + (ic + 0) % ndiv_circumference;
            let i2 = (ih + 0) * ndiv_circumference + 1 + (ic + 1) % ndiv_circumference;
            let i3 = (ih + 1) * ndiv_circumference + 1 + (ic + 1) % ndiv_circumference;
            let i4 = (ih + 1) * ndiv_circumference + 1 + (ic + 0) % ndiv_circumference;
            tri2vtx.push([i3.as_(), i2.as_(), i1.as_()]);
            tri2vtx.push([i4.as_(), i3.as_(), i1.as_()]);
        }
    }
    for ic in 0..ndiv_circumference {
        let i0 = ndiv_circumference * (ndiv_longtitude - 1) + 1;
        let i1 = (ndiv_longtitude - 2) * ndiv_circumference + 1 + (ic + 1) % ndiv_circumference;
        let i2 = (ndiv_longtitude - 2) * ndiv_circumference + 1 + (ic + 0) % ndiv_circumference;
        tri2vtx.push([i0.as_(), i1.as_(), i2.as_()]);
    }
    tri2vtx
}

/// generate 3D mesh of closed cylinder
/// * `radius` - radius
/// * 'length' - length
#[allow(clippy::identity_op)]
pub fn cylinder_closed_end_yup<T>(
    radius: T,
    length: T,
    ndiv_circumference: usize,
    ndiv_length: usize,
    is_center: bool,
) -> (Vec<[usize; 3]>, Vec<[T; 3]>)
where
    T: num_traits::FloatConst + 'static + Copy + num_traits::Float,
    usize: AsPrimitive<T>,
{
    let num_vtx = ndiv_circumference * (ndiv_length + 1) + 2;
    let mut vtx2xyz = vec![[T::zero(); 3]; num_vtx];
    assert!(ndiv_length >= 1);
    assert!(ndiv_circumference > 2);
    let zero = T::zero();
    let two = T::one() + T::one();
    let half = T::one() / two;
    let pi: T = T::PI();
    let dl: T = length / ndiv_length.as_();
    let dr: T = two * pi / ndiv_circumference.as_();
    let y_min = if is_center { -length * half } else { zero };
    // bottom
    vtx2xyz[0] = [zero, y_min, zero];
    for il in 0..ndiv_length + 1 {
        let y0 = y_min + dl * il.as_();
        for ilo in 0..ndiv_circumference {
            let x0 = radius * (dr * ilo.as_()).cos();
            let z0 = radius * (dr * ilo.as_()).sin();
            let i_vtx = il * ndiv_circumference + ilo + 1;
            vtx2xyz[i_vtx] = [x0, y0, z0];
        }
    }
    // top
    vtx2xyz[num_vtx - 1] = [zero, y_min + length, zero];
    // ------------------------------------
    let tri2vtx = cylinder_closed_end_topology::<usize>(ndiv_length, ndiv_circumference);
    //let tri2vtx = nalgebra::Matrix3xX::<usize>::from_column_slice(&tri2vtx);
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_cylider_closed_end_tri3() {
    cylinder_closed_end_yup::<f32>(1., 1., 32, 32, true);
    cylinder_closed_end_yup::<f64>(1., 1., 32, 32, true);
}

// ------------------------

#[allow(clippy::identity_op)]
pub fn capsule_yup<T>(
    r: T,
    l: T,
    ndiv_circum: usize,
    ndiv_longtitude: usize,
    ndiv_length: usize,
) -> (Vec<[usize; 3]>, Vec<[T; 3]>)
where
    T: num_traits::FloatConst + 'static + Copy + num_traits::Float,
    usize: AsPrimitive<T>,
{
    let (tri2vtx, mut vtx2xyz) = cylinder_closed_end_yup::<T>(
        T::one(),
        T::one(),
        ndiv_circum,
        2 * ndiv_longtitude + ndiv_length - 2,
        true,
    );
    assert_eq!(
        vtx2xyz.len(),
        (2 * ndiv_longtitude + ndiv_length - 1) * ndiv_circum + 2
    );
    let pi: T = T::PI();
    let one = T::one();
    let half: T = one / (one + one);
    {
        // South Pole
        vtx2xyz[0][0] = T::zero();
        vtx2xyz[0][1] = -l * half - r;
        vtx2xyz[0][2] = T::zero();
    }
    for ir in 0..ndiv_longtitude {
        let t0 = pi * half * (ndiv_longtitude - 1 - ir).as_() / ndiv_longtitude.as_();
        let y0 = -l * half - r * t0.sin();
        let c0 = r * t0.cos();
        for ic in 0..ndiv_circum {
            let theta = 2.as_() * pi * ic.as_() / ndiv_circum.as_();
            vtx2xyz[1 + ir * ndiv_circum + ic][0] = c0 * theta.cos();
            vtx2xyz[1 + ir * ndiv_circum + ic][1] = y0;
            vtx2xyz[1 + ir * ndiv_circum + ic][2] = c0 * theta.sin();
        }
    }
    for il in 0..ndiv_length - 1 {
        let y0 = -l * half + (il + 1).as_() * l / ndiv_length.as_();
        for ic in 0..ndiv_circum {
            let theta = 2.as_() * pi * ic.as_() / ndiv_circum.as_();
            vtx2xyz[1 + (il + ndiv_longtitude) * ndiv_circum + ic][0] = r * theta.cos();
            vtx2xyz[1 + (il + ndiv_longtitude) * ndiv_circum + ic][1] = y0;
            vtx2xyz[1 + (il + ndiv_longtitude) * ndiv_circum + ic][2] = r * theta.sin();
        }
    }
    for ir in 0..ndiv_longtitude {
        let t0 = pi * half * ir.as_() / ndiv_longtitude.as_();
        let y0 = l * half + r * t0.sin();
        let c0 = r * t0.cos();
        for ic in 0..ndiv_circum {
            let theta = 2.as_() * pi * ic.as_() / ndiv_circum.as_();
            vtx2xyz[1 + (ir + ndiv_length + ndiv_longtitude - 1) * ndiv_circum + ic][0] =
                c0 * theta.cos();
            vtx2xyz[1 + (ir + ndiv_length + ndiv_longtitude - 1) * ndiv_circum + ic][1] = y0;
            vtx2xyz[1 + (ir + ndiv_length + ndiv_longtitude - 1) * ndiv_circum + ic][2] =
                c0 * theta.sin();
        }
    }
    {
        // North Pole
        let np = vtx2xyz.len() / 3;
        vtx2xyz[np - 1][0] = T::zero();
        vtx2xyz[np - 1][1] = l * half + r;
        vtx2xyz[np - 1][2] = T::zero();
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_capsule_tri3() {
    capsule_yup::<f32>(1., 1., 32, 12, 5);
    capsule_yup::<f64>(1., 1., 32, 12, 5);
}

pub fn capsule_connecting_two_point<T>(
    p0: &[T; 3],
    p1: &[T; 3],
    rad: T,
    ndiv_circum: usize,
    ndiv_longtitude: usize,
    ndiv_length: usize,
) -> (Vec<[usize; 3]>, Vec<[T; 3]>)
where
    T: Copy + num_traits::Float + num_traits::FloatConst + 'static,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let one = T::one();
    let half = one / (one + one);
    let len = p1.sub(p0).norm();
    let (tri2vtx, mut vtx2xyz) = capsule_yup(rad, len, ndiv_circum, ndiv_longtitude, ndiv_length);
    let q2 = [T::zero(), len * half, T::zero()];
    let mat = del_geo_core::mat3_col_major::minimum_rotation_matrix(
        &[T::zero(), T::one(), T::zero()],
        &p1.sub(p0).normalize(),
    );
    for p in vtx2xyz.iter_mut() {
        let q = del_geo_core::mat3_col_major::mult_vec(&mat, &q2.add(p)).add(p0);
        p.copy_from_slice(&q);
    }
    (tri2vtx, vtx2xyz)
}

// --------------------------------------------------------

#[allow(clippy::identity_op)]
pub fn torus_zup<Index, Float>(
    major_radius: Float,
    minor_radius: Float,
    ndiv_major: usize, // latitude
    ndiv_minor: usize,
) -> (Vec<[Index; 3]>, Vec<[Float; 3]>)
where
    Float: num_traits::Float + Default + 'static,
    Index: num_traits::PrimInt + 'static + Copy,
    f32: AsPrimitive<Float>,
    usize: AsPrimitive<Float> + AsPrimitive<Index>,
{
    let rlg: Float = (std::f32::consts::PI * 2_f32).as_() / ndiv_major.as_(); // latitude
    let rlt: Float = (std::f32::consts::PI * 2_f32).as_() / ndiv_minor.as_();
    let mut vtx2xyz: Vec<_> = vec![[Float::zero(); 3]; ndiv_major * ndiv_minor];
    for ilg in 0..ndiv_major {
        for ilt in 0..ndiv_minor {
            let lt: Float = <usize as AsPrimitive<Float>>::as_(ilt) * rlt;
            let lg: Float = <usize as AsPrimitive<Float>>::as_(ilg) * rlg;
            let r0: Float = major_radius + minor_radius * lt.cos();
            vtx2xyz[ilg * ndiv_minor + ilt][0] = r0 * lg.sin();
            vtx2xyz[ilg * ndiv_minor + ilt][1] = r0 * lg.cos();
            vtx2xyz[ilg * ndiv_minor + ilt][2] = minor_radius * lt.sin();
        }
    }
    let mut tri2vtx: Vec<[Index; 3]> = vec![[Index::zero(); 3]; ndiv_major * ndiv_minor * 2];
    for ilg in 0..ndiv_major {
        for ilt in 0..ndiv_minor {
            let iug = if ilg == ndiv_major - 1 { 0 } else { ilg + 1 };
            let iut = if ilt == ndiv_minor - 1 { 0 } else { ilt + 1 };
            tri2vtx[(ilg * ndiv_minor + ilt) * 2][0] = (ilg * ndiv_minor + ilt).as_();
            tri2vtx[(ilg * ndiv_minor + ilt) * 2][2] = (iug * ndiv_minor + ilt).as_();
            tri2vtx[(ilg * ndiv_minor + ilt) * 2][1] = (iug * ndiv_minor + iut).as_();
            tri2vtx[(ilg * ndiv_minor + ilt) * 2 + 1][0] = (ilg * ndiv_minor + ilt).as_();
            tri2vtx[(ilg * ndiv_minor + ilt) * 2 + 1][2] = (iug * ndiv_minor + iut).as_();
            tri2vtx[(ilg * ndiv_minor + ilt) * 2 + 1][1] = (ilg * ndiv_minor + iut).as_();
        }
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_torus_tri3() {
    torus_zup::<usize, f32>(1., 1., 32, 32);
    torus_zup::<usize, f64>(1., 1., 32, 32);
}

// --------------

/// the pole is y-axis
#[allow(clippy::identity_op)]
pub fn sphere_yup<Index, Real>(
    radius: Real,
    n_longitude: usize,
    n_latitude: usize,
) -> crate::trimesh3::TriMesh3Owned<Index, Real>
where
    Real: num_traits::Float + 'static,
    Index: num_traits::PrimInt + 'static,
    f32: AsPrimitive<Real>,
    usize: AsPrimitive<Real> + AsPrimitive<Index>,
{
    let mut vtx2xyz = Vec::<[Real; 3]>::new();
    let mut tri2vtx = Vec::<[Index; 3]>::new();
    vtx2xyz.clear();
    if n_longitude <= 1 || n_latitude <= 2 {
        return TriMesh3Owned { tri2vtx, vtx2xyz };
    }
    let pi: Real = std::f32::consts::PI.as_();
    let dl: Real = pi / n_longitude.as_();
    let two = Real::one() + Real::one();
    let dr: Real = two * pi / n_latitude.as_();
    vtx2xyz.reserve((n_latitude * (n_longitude - 1) + 2) * 3);
    for ila in 0..n_longitude + 1 {
        let y0 = (dl * ila.as_()).cos();
        let r0 = (dl * ila.as_()).sin();
        for ilo in 0..n_latitude {
            let x0 = r0 * (dr * ilo.as_()).sin();
            let z0 = r0 * (dr * ilo.as_()).cos();
            vtx2xyz.push([radius * x0, radius * y0, radius * z0]);
            if ila == 0 || ila == n_longitude {
                break;
            }
        }
    }
    //
    let ntri = n_latitude * (n_longitude - 1) * 2 + n_latitude * 2;
    tri2vtx.reserve(ntri * 3);

    let tri2vtx = cylinder_closed_end_topology::<Index>(n_longitude - 2, n_latitude);
    TriMesh3Owned { tri2vtx, vtx2xyz }
}

#[test]
fn test_sphere_yup() {
    let path_dir = std::path::Path::new("../target/out_del_msh_cpu");
    std::fs::create_dir_all(path_dir).unwrap();
    let trimesh3 = sphere_yup::<usize, f64>(1.0, 16, 8);
    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(
        path_dir.join("trimesh3_primitive_sphere_yup.obj"),
        &trimesh3.tri2vtx,
        &trimesh3.vtx2xyz,
    )
    .unwrap();
}

// ----------------------------------------

#[allow(clippy::identity_op)]
pub fn hemisphere_zup<T>(
    radius: T,
    n_longitude: usize,
    n_latitude: usize,
) -> (Vec<[usize; 3]>, Vec<[T; 3]>)
where
    T: num_traits::Float + 'static,
    f32: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    if n_longitude == 0 || n_latitude <= 2 {
        return (vec![], vec![]);
    }
    let pi: T = std::f32::consts::PI.as_();
    let dl: T = 0.5.as_() * pi / n_longitude.as_();
    let dr: T = 2.as_() * pi / n_latitude.as_();
    let nvtx = n_latitude * n_longitude + 1;
    let mut vtx2xyz = Vec::<[T; 3]>::with_capacity(nvtx);
    for ila in 0..n_longitude + 1 {
        let z0 = (dl * ila.as_()).cos();
        let r0 = (dl * ila.as_()).sin();
        for ilo in 0..n_latitude {
            let x0 = r0 * (dr * ilo.as_()).sin();
            let y0 = r0 * (dr * ilo.as_()).cos();
            vtx2xyz.push([radius * x0, radius * y0, radius * z0]);
            if ila == 0 {
                break;
            }
        }
    }
    assert_eq!(nvtx, vtx2xyz.len());
    //
    let ntri = n_latitude * (n_longitude - 1) * 2 + n_latitude;
    let mut tri2vtx = Vec::<[usize; 3]>::with_capacity(ntri);
    for ilo in 0..n_latitude {
        tri2vtx.push([0, (ilo + 0) % n_latitude + 1, (ilo + 1) % n_latitude + 1]);
    }
    for ilong in 0..n_longitude - 1 {
        for ilat in 0..n_latitude {
            let i1 = (ilong + 0) * n_latitude + 1 + (ilat + 0) % n_latitude;
            let i2 = (ilong + 0) * n_latitude + 1 + (ilat + 1) % n_latitude;
            let i3 = (ilong + 1) * n_latitude + 1 + (ilat + 1) % n_latitude;
            let i4 = (ilong + 1) * n_latitude + 1 + (ilat + 0) % n_latitude;
            tri2vtx.push([i3, i2, i1]);
            tri2vtx.push([i4, i3, i1]);
        }
    }
    assert_eq!(ntri, tri2vtx.len());
    (tri2vtx, vtx2xyz)
}

// ---------------------------

#[allow(clippy::identity_op)]
pub fn bypyramid_zup<Real>(
    length: Real,
    rad_ratio: Real,
    node_ratio: Real,
) -> (Vec<[usize; 3]>, Vec<[Real; 3]>)
where
    Real: num_traits::FloatConst + num_traits::Float + 'static,
    usize: AsPrimitive<Real>,
{
    let zero = Real::zero();
    let mut vtx2xyz: Vec<[Real; 3]> = vec![[zero, zero, zero], [zero, zero, length]];
    {
        let dt = Real::PI() / (Real::one() + Real::one());
        let r0 = length * rad_ratio;
        for idiv in 0..4 {
            let s0 = r0 * (idiv.as_() * dt).sin();
            let c0 = r0 * (idiv.as_() * dt).cos();
            vtx2xyz.push([s0, c0, length * node_ratio]);
        }
    }
    //
    let mut tri2vtx: Vec<[usize; 3]> = vec![];
    for idiv in 0..4 {
        tri2vtx.push([0, 2 + (0 + idiv) % 4, 2 + (1 + idiv) % 4]);
        //
        tri2vtx.push([1, 2 + (1 + idiv) % 4, 2 + (0 + idiv) % 4]);
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_biypyramid_zup() {
    let (tri2vtx, vtx2xyz) = bypyramid_zup::<f64>(2.0, 0.2, 0.3);
    let path_out = std::path::Path::new("../target/out_del_msh_cpu/bipyramid_zup.obj");
    std::fs::create_dir_all(path_out.parent().unwrap()).unwrap();
    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(path_out, &tri2vtx, &vtx2xyz).unwrap();
}

// ------------------

#[allow(clippy::identity_op)]
fn arrow_yup<Real>(num_division_circumference: usize) -> (Vec<[usize; 3]>, Vec<[Real; 3]>)
where
    Real: num_traits::Float + num_traits::FloatConst + 'static + Copy,
    usize: AsPrimitive<Real>,
{
    let one = Real::one();
    let two = one + one;
    let three = two + one;
    let five = one + two + two;
    let dr: Real = Real::PI() * two / num_division_circumference.as_();
    let stem_height: Real = two / three;
    let radius_small: Real = one / (five * two);
    let radius_large: Real = one / five;
    let (tri2vtx, mut vtx2xyz) = cylinder_closed_end_yup(
        Real::one(),
        Real::one(),
        num_division_circumference,
        2,
        true,
    );
    assert_eq!(vtx2xyz.len(), 2 + 3 * num_division_circumference);
    vtx2xyz[0][0] = Real::zero();
    vtx2xyz[0][1] = Real::zero();
    vtx2xyz[0][2] = Real::zero();
    let height_rad = [
        (Real::zero(), radius_small),
        (stem_height, radius_small),
        (stem_height, radius_large),
    ];
    for (il, (height, rad)) in height_rad.iter().enumerate() {
        for ilo in 0..num_division_circumference {
            let theta: Real = dr * ilo.as_();
            let x0 = theta.cos() * *rad;
            let z0 = theta.sin() * *rad;
            vtx2xyz[1 + il * num_division_circumference + ilo][0] = x0;
            vtx2xyz[1 + il * num_division_circumference + ilo][1] = *height;
            vtx2xyz[1 + il * num_division_circumference + ilo][2] = z0;
        }
    }
    let n = 1 + 3 * num_division_circumference;
    vtx2xyz[n][0] = Real::zero();
    vtx2xyz[n][1] = Real::one();
    vtx2xyz[n][2] = Real::zero();
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_arrow_zup() {
    let (tri2vtx, vtx2xyz) = arrow_yup::<f64>(16);
    let path = std::path::Path::new("../target/out_del_msh_cpu/arrow_zup.obj");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(path, &tri2vtx, &vtx2xyz).unwrap();
}

// --------------------------------

pub fn arrow_connecting_two_points<T>(
    p0: &[T; 3],
    p1: &[T; 3],
    num_division_circumference: usize,
) -> (Vec<[usize; 3]>, Vec<[T; 3]>)
where
    T: Copy + num_traits::Float + num_traits::FloatConst + 'static,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vec3::Vec3;
    let len = p1.sub(p0).norm();
    let (tri2vtx, mut vtx2xyz) = arrow_yup(num_division_circumference);
    let mat = del_geo_core::mat3_col_major::minimum_rotation_matrix(
        &[T::zero(), T::one(), T::zero()],
        &p1.sub(p0).normalize(),
    );
    let mat = del_geo_core::mat3_col_major::scale(&mat, len);
    for v in vtx2xyz.iter_mut() {
        let q1 = del_geo_core::mat3_col_major::mult_vec(&mat, v).add(p0);
        v.copy_from_slice(&q1);
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_arrow_connecting_two_points() {
    let (tri2vtx, vtx2xyz) =
        arrow_connecting_two_points::<f64>(&[1.0, 1.0, 1.0], &[1.0, 1.0, 2.0], 16);
    let path = std::path::Path::new("../target/out_del_msh_cpu/arrow_connecting_two_points.obj");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(path, &tri2vtx, &vtx2xyz).unwrap();
}

// ------------------------------

pub fn obb3<Real>(obb: &[Real; 12]) -> (Vec<[usize; 3]>, Vec<[Real; 3]>)
where
    Real: num_traits::Float,
{
    let vtx2xyz: Vec<[Real; 3]> = del_geo_core::obb3::corner_points(obb).to_vec();
    let tri2vtx: Vec<[usize; 3]> = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    (tri2vtx, vtx2xyz)
}

// -----------------------------

pub fn annulus_yup<Real>(
    r_small: Real,
    r_large: Real,
    ndiv_radius: usize,
    ndiv_theta: usize,
) -> (Vec<[usize; 3]>, Vec<[Real; 3]>)
where
    Real: num_traits::Float + num_traits::FloatConst + 'static,
    usize: AsPrimitive<Real>,
{
    let zero = Real::zero();
    let one = Real::one();
    let two = one + one;
    let half = one / two;
    let mut vtx2xyz = Vec::<[Real; 3]>::with_capacity((ndiv_radius + 1) * ndiv_theta);
    {
        // make coordinates
        let dr = (r_large - r_small) / ndiv_radius.as_();
        let dth = two * Real::PI() / ndiv_theta.as_();
        for ir in 0..=ndiv_radius {
            for ith in 0..ndiv_theta {
                let rad = dr * ir.as_() + r_small;
                let theta = (ith.as_() + (ir % 2).as_() * half) * dth;
                vtx2xyz.push([rad * theta.cos(), zero, rad * theta.sin()]);
            }
        }
    }

    let mut tri2vtx = Vec::<[usize; 3]>::with_capacity(ndiv_radius * ndiv_theta * 2);
    for ir in 0..ndiv_radius {
        #[allow(clippy::identity_op)]
        for ith in 0..ndiv_theta {
            let i1 = (ir + 0) * ndiv_theta + (ith + 0) % ndiv_theta;
            let i2 = (ir + 0) * ndiv_theta + (ith + 1) % ndiv_theta;
            let i3 = (ir + 1) * ndiv_theta + (ith + 1) % ndiv_theta;
            let i4 = (ir + 1) * ndiv_theta + (ith + 0) % ndiv_theta;
            if ir % 2 == 1 {
                tri2vtx.push([i3, i1, i2]);
                tri2vtx.push([i4, i1, i3]);
            } else {
                tri2vtx.push([i4, i2, i3]);
                tri2vtx.push([i4, i1, i2]);
            }
        }
    }
    (tri2vtx, vtx2xyz)
}

#[test]
fn test_annulus_yup() {
    let (tri2vtx, vtx2xyz) = annulus_yup(0.3, 0.8, 32, 64);
    let path = std::path::Path::new("../target/out_del_msh_cpu/annulus.obj");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(path, &tri2vtx, &vtx2xyz).unwrap();
}
