//! methods for 3D triangle mesh

use num_traits::AsPrimitive;

pub fn vtx2normal(tri2vtx: &[usize], vtx2xyz: &[f64]) -> Vec<f64> {
    let mut vtx2nrm = vec![0_f64; vtx2xyz.len()];
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref!(vtx2xyz, i0 * 3, 3);
        let p1 = arrayref::array_ref!(vtx2xyz, i1 * 3, 3);
        let p2 = arrayref::array_ref!(vtx2xyz, i2 * 3, 3);
        let (un, _area) = del_geo_core::tri3::unit_normal_area(p0, p1, p2);
        for &i_vtx in &node2vtx[0..3] {
            vtx2nrm[i_vtx * 3] += un[0];
            vtx2nrm[i_vtx * 3 + 1] += un[1];
            vtx2nrm[i_vtx * 3 + 2] += un[2];
        }
    }
    for v in vtx2nrm.chunks_mut(3) {
        del_geo_core::vec3::normalize_in_place(v.try_into().unwrap());
    }
    vtx2nrm
}

pub fn vtx2area<T>(tri2vtx: &[usize], vtx2xyz: &[T]) -> Vec<T>
where
    T: num_traits::Float + std::ops::AddAssign + std::ops::MulAssign,
{
    let num_vtx = vtx2xyz.len() / 3;
    let mut areas = vec![T::zero(); num_vtx];
    let one_third = T::one() / (T::one() + T::one() + T::one());
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref!(vtx2xyz, i0 * 3, 3);
        let p1 = arrayref::array_ref!(vtx2xyz, i1 * 3, 3);
        let p2 = arrayref::array_ref!(vtx2xyz, i2 * 3, 3);
        let a0 = del_geo_core::tri3::area(p0, p1, p2) * one_third;
        areas[i0] += a0;
        areas[i1] += a0;
        areas[i2] += a0;
    }
    areas
}

#[test]
fn test_vtx2area() {
    let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::sphere_yup(1_f64, 128, 256);
    let vtx2area = crate::trimesh3::vtx2area(&tri2vtx, &vtx2xyz);
    let total_area: f64 = vtx2area.iter().sum();
    assert!((total_area - std::f64::consts::PI * 4.0).abs() < 1.0e-2);
}

pub fn to_corner_points<Index, Real>(
    tri2vtx: &[Index],
    vtx2xyz: &[Real],
    i_tri: usize,
) -> ([Real; 3], [Real; 3], [Real; 3])
where
    Real: Copy,
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    let i0: usize = tri2vtx[i_tri * 3].as_();
    let i1: usize = tri2vtx[i_tri * 3 + 1].as_();
    let i2: usize = tri2vtx[i_tri * 3 + 2].as_();
    (
        [vtx2xyz[i0 * 3], vtx2xyz[i0 * 3 + 1], vtx2xyz[i0 * 3 + 2]],
        [vtx2xyz[i1 * 3], vtx2xyz[i1 * 3 + 1], vtx2xyz[i1 * 3 + 2]],
        [vtx2xyz[i2 * 3], vtx2xyz[i2 * 3 + 1], vtx2xyz[i2 * 3 + 2]],
    )
}

// above: vtx2*** method
// -------------------------

pub fn tri2normal<T, U>(tri2vtx: &[U], vtx2xyz: &[T]) -> Vec<T>
where
    T: num_traits::Float,
    U: AsPrimitive<usize>,
{
    let mut tri2normal = Vec::<T>::with_capacity(tri2vtx.len());
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0].as_(), node2vtx[1].as_(), node2vtx[2].as_());
        let n = del_geo_core::tri3::normal(
            arrayref::array_ref!(vtx2xyz, i0 * 3, 3),
            arrayref::array_ref!(vtx2xyz, i1 * 3, 3),
            arrayref::array_ref!(vtx2xyz, i2 * 3, 3),
        );
        tri2normal.extend_from_slice(&n);
    }
    tri2normal
}

pub fn tri2area(tri2vtx: &[usize], vtx2xyz: &[f32]) -> Vec<f32> {
    let num_tri = tri2vtx.len() / 3;
    let mut tri2area = Vec::<f32>::with_capacity(num_tri);
    for i_tri in 0..num_tri {
        let area = to_tri3(tri2vtx, vtx2xyz, i_tri).area();
        tri2area.push(area);
    }
    tri2area
}

// above: elem2*** methods
// ---------------------------

pub fn extend_avoid_intersection(
    tri2vtx: &[usize],
    vtx2xyz: &[f64],
    q: &[f64],
    step: f64,
) -> [f64; 3] {
    let q = nalgebra::Vector3::<f64>::from_row_slice(q);
    let mut dq = nalgebra::Vector3::<f64>::zeros();
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[i0 * 3..i0 * 3 + 3]);
        let p1 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[i1 * 3..i1 * 3 + 3]);
        let p2 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[i2 * 3..i2 * 3 + 3]);
        let (_, dw) =
            del_geo_nalgebra::tri3::wdw_integral_of_inverse_distance_cubic(&p0, &p1, &p2, &q);
        dq -= dw;
    }
    let q = q + dq.normalize() * step;
    [q[0], q[1], q[2]]
}

pub fn mean_edge_length(tri2vtx: &[usize], vtx2xyz: &[f32]) -> f32 {
    let num_tri = tri2vtx.len() / 3;
    let mut sum = 0_f32;
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref![vtx2xyz, i0 * 3, 3];
        let p1 = arrayref::array_ref![vtx2xyz, i1 * 3, 3];
        let p2 = arrayref::array_ref![vtx2xyz, i2 * 3, 3];
        sum += del_geo_core::vec3::distance(p0, p1);
        sum += del_geo_core::vec3::distance(p1, p2);
        sum += del_geo_core::vec3::distance(p2, p0);
    }
    sum / (num_tri * 3) as f32
}

// --------------

/// find the nearest point on the 3D triangle mesh (`tri2vtx`, `vtx2xyz`) from input point (`q`)
pub fn distance_to_point3(tri2vtx: &[usize], vtx2xyz: &[f32], q: [f32; 3]) -> f32 {
    let q = nalgebra::Vector3::new(q[0], q[1], q[2]);
    let mut dist_min = f32::MAX;
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = crate::vtx2xyz::to_navec3(vtx2xyz, i0);
        let p1 = crate::vtx2xyz::to_navec3(vtx2xyz, i1);
        let p2 = crate::vtx2xyz::to_navec3(vtx2xyz, i2);
        let (p012, _r0, _r1) = del_geo_nalgebra::tri3::nearest_to_point3(&p0, &p1, &p2, &q);
        let dist = (p012 - q).norm();
        if dist_min > dist {
            dist_min = dist;
        }
    }
    dist_min
}

pub fn distance_to_points3(tri2vtx: &[usize], vtx2xyz: &[f32], hv2xyz: &[f32]) -> f32 {
    let mut max_dist = 0f32;
    for i_hv in 0..hv2xyz.len() / 3 {
        let min_dist = crate::trimesh3::distance_to_point3(
            tri2vtx,
            vtx2xyz,
            hv2xyz[i_hv * 3..i_hv * 3 + 3].try_into().unwrap(),
        );
        if max_dist < min_dist {
            max_dist = min_dist;
        }
    }
    max_dist
}

pub fn area(tri2vtx: &[usize], vtx2xyz: &[f32]) -> f32 {
    let mut sum_area = 0f32;
    for i_tri in 0..tri2vtx.len() / 3 {
        sum_area += to_tri3(tri2vtx, vtx2xyz, i_tri).area();
    }
    sum_area
}

pub fn cog_and_area(tri2vtx: &[usize], vtx2xyz: &[f32]) -> Option<([f32; 3], f32)> {
    use del_geo_core::vec3;
    let mut sum_area = 0f32;
    let mut sum_cg = [0f32; 3];
    for i_tri in 0..tri2vtx.len() / 3 {
        let tri = to_tri3(tri2vtx, vtx2xyz, i_tri);
        let area = tri.area();
        sum_area += area;
        sum_cg = vec3::add(&sum_cg, &vec3::scale(&tri.cog(), area));
    }
    if sum_area == 0f32 {
        return None;
    }
    let sum_cog = vec3::scale(&sum_cg, 1.0 / sum_area);
    Some((sum_cog, sum_area))
}

// ---------------------

pub fn to_tri3<'a, Index, Real>(
    tri2vtx: &'a [Index],
    vtx2xyz: &'a [Real],
    i_tri: usize,
) -> del_geo_core::tri3::Tri3<'a, Real>
where
    Index: AsPrimitive<usize>,
{
    let i0: usize = tri2vtx[i_tri * 3].as_();
    let i1: usize = tri2vtx[i_tri * 3 + 1].as_();
    let i2: usize = tri2vtx[i_tri * 3 + 2].as_();
    del_geo_core::tri3::Tri3 {
        p0: arrayref::array_ref!(vtx2xyz, i0 * 3, 3),
        p1: arrayref::array_ref!(vtx2xyz, i1 * 3, 3),
        p2: arrayref::array_ref!(vtx2xyz, i2 * 3, 3),
    }
}
