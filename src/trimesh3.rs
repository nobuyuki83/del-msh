//! methods for 3D triangle mesh

use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn vtx2normal(
    tri2vtx: &[usize],
    vtx2xyz: &[f64]) -> Vec<f64> {
    let mut vtx2nrm = vec!(0_f64; vtx2xyz.len());
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = &vtx2xyz[i0 * 3..i0 * 3 + 3].try_into().unwrap();
        let p1 = &vtx2xyz[i1 * 3..i1 * 3 + 3].try_into().unwrap();
        let p2 = &vtx2xyz[i2 * 3..i2 * 3 + 3].try_into().unwrap();
        let (un, _area) = del_geo::tri3::unit_normal_area_(p0, p1, p2);
        for &i_vtx in &node2vtx[0..3] {
            vtx2nrm[i_vtx * 3 + 0] += un[0];
            vtx2nrm[i_vtx * 3 + 1] += un[1];
            vtx2nrm[i_vtx * 3 + 2] += un[2];
        }
    }
    for v in vtx2nrm.chunks_mut(3) {
        del_geo::vec3::normalize_(v.try_into().unwrap());
    }
    vtx2nrm
}

#[allow(clippy::identity_op)]
pub fn vtx2area<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T]) -> Vec<T>
    where T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
          f64: AsPrimitive<T>
{
    let num_vtx = vtx2xyz.len() / 3;
    let mut areas = vec!(T::zero(); num_vtx);
    let one_third = T::one() / 3_f64.as_();
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = &vtx2xyz[i0 * 3..i0 * 3 + 3].try_into().unwrap();
        let p1 = &vtx2xyz[i1 * 3..i1 * 3 + 3].try_into().unwrap();
        let p2 = &vtx2xyz[i2 * 3..i2 * 3 + 3].try_into().unwrap();
        let a0 = del_geo::tri3::area_(p0, p1, p2) * one_third;
        areas[i0] += a0;
        areas[i1] += a0;
        areas[i2] += a0;
    }
    areas
}


#[cfg(test)]
mod tests {
    use num_traits::FloatConst;
    #[test]
    fn test_vtx2area() {
        let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::sphere_yup(
            1_f64, 128, 256);
        let vtx2area = crate::trimesh3::vtx2area(&tri2vtx, &vtx2xyz);
        let total_area: f64 = vtx2area.iter().sum();
        assert!((total_area - f64::PI() * 4.0).abs() < 1.0e-2);
    }
}

// above: vtx2*** method
// -------------------------

#[allow(clippy::identity_op)]
pub fn tri2normal<T, U>(
    tri2vtx: &[U],
    vtx2xyz: &[T]) -> Vec<T>
where T: num_traits::Float,
      U: num_traits::AsPrimitive<usize>
{
    let mut tri2normal = Vec::<T>::with_capacity(tri2vtx.len() );
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0].as_(), node2vtx[1].as_(), node2vtx[2].as_());
        let n = del_geo::tri3::normal_(
            &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3].try_into().unwrap(),
            &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3].try_into().unwrap(),
            &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3].try_into().unwrap());
        tri2normal.extend_from_slice(&n);
    }
    tri2normal
}

#[allow(clippy::identity_op)]
pub fn tri2area(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> Vec<f32>
{
    let mut tri2area = Vec::<f32>::with_capacity(tri2vtx.len() / 3);
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let area = del_geo::tri3::area_(
            &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3].try_into().unwrap(),
            &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3].try_into().unwrap(),
            &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3].try_into().unwrap());
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
    step: f64) -> [f64; 3] {
    let q = nalgebra::Vector3::<f64>::from_row_slice(q);
    let mut dq = nalgebra::Vector3::<f64>::zeros();
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[i0 * 3..i0 * 3 + 3]);
        let p1 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[i1 * 3..i1 * 3 + 3]);
        let p2 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[i2 * 3..i2 * 3 + 3]);
        let (_, dw) = del_geo::tri3::wdw_integral_of_inverse_distance_cubic(&p0, &p1, &p2, &q);
        dq -= dw;
    }
    let q = q + dq.normalize() * step;
    [q[0], q[1], q[2]]
}



#[allow(clippy::identity_op)]
pub fn mean_edge_length(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> f32 {
    let num_tri = tri2vtx.len() / 3;
    let mut sum = 0_f32;
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = &vtx2xyz[i0 * 3..i0 * 3 + 3].try_into().unwrap();
        let p1 = &vtx2xyz[i1 * 3..i1 * 3 + 3].try_into().unwrap();
        let p2 = &vtx2xyz[i2 * 3..i2 * 3 + 3].try_into().unwrap();
        sum += del_geo::vec3::distance_(p0, p1);
        sum += del_geo::vec3::distance_(p1, p2);
        sum += del_geo::vec3::distance_(p2, p0);
    }
    sum / (num_tri * 3) as f32
}




