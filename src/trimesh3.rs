//! methods for 3D triangle mesh

use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn vtx2normal(
    tri2vtx: &[usize],
    vtx2xyz: &[f64]) -> Vec<f64> {
    let mut vtx2nrm = vec!(0_f64; vtx2xyz.len());
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = &vtx2xyz[i0 * 3..i0 * 3 + 3];
        let p1 = &vtx2xyz[i1 * 3..i1 * 3 + 3];
        let p2 = &vtx2xyz[i2 * 3..i2 * 3 + 3];
        let mut un = [0_f64; 3];
        del_geo::tri3::unit_normal_(&mut un, p0, p1, p2);
        for &i_vtx in &node2vtx[0..3] {
            vtx2nrm[i_vtx * 3 + 0] += un[0];
            vtx2nrm[i_vtx * 3 + 1] += un[1];
            vtx2nrm[i_vtx * 3 + 2] += un[2];
        }
    }
    for v in vtx2nrm.chunks_mut(3) {
        del_geo::vec3::normalize_(v);
    }
    vtx2nrm
}

pub fn extend_avoid_intersection(
    tri2vtx: &[usize],
    vtx2xyz: &[f64],
    q: &[f64],
    step: f64) -> [f64; 3] {
    let q = nalgebra::Vector3::<f64>::from_row_slice(q);
    let mut dq = nalgebra::Vector3::<f64>::zeros();
    for t in tri2vtx.chunks(3) {
        let p0 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[t[0] * 3..t[0] * 3 + 3]);
        let p1 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[t[1] * 3..t[1] * 3 + 3]);
        let p2 = nalgebra::Vector3::<f64>::from_row_slice(&vtx2xyz[t[2] * 3..t[2] * 3 + 3]);
        let (_, dw) = del_geo::tri3::wdw_integral_of_inverse_distance_cubic(&p0, &p1, &p2, &q);
        dq -= dw;
    }
    let q = q + dq.normalize() * step;
    [q[0], q[1], q[2]]
}

#[allow(clippy::identity_op)]
pub fn elem2area(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> Vec<f32>
{
    let mut tri2area = Vec::<f32>::with_capacity(tri2vtx.len() / 3);
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let area = del_geo::tri3::area_(
            &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3],
            &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3],
            &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3]);
        tri2area.push(area);
    }
    tri2area
}

#[allow(clippy::identity_op)]
pub fn vtx2area<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T]) -> Vec<T>
    where T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
          f64: AsPrimitive<T>
{
    let num_vtx = vtx2xyz.len() / 3;
    let num_tri = tri2vtx.len() / 3;
    let mut areas = vec!(T::zero(); num_vtx);
    let one_third = T::one() / 3_f64.as_();
    for i_tri in 0..num_tri {
        let i0 = tri2vtx[i_tri * 3 + 0];
        let i1 = tri2vtx[i_tri * 3 + 1];
        let i2 = tri2vtx[i_tri * 3 + 2];
        let a0 = del_geo::tri3::area_(
            &vtx2xyz[i0 * 3..i0 * 3 + 3],
            &vtx2xyz[i1 * 3..i1 * 3 + 3],
            &vtx2xyz[i2 * 3..i2 * 3 + 3]) * one_third;
        areas[i0] += a0;
        areas[i1] += a0;
        areas[i2] += a0;
    }
    areas
}

#[allow(clippy::identity_op)]
pub fn position_from_barycentric_coordinate<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    itri: usize,
    r0: T,
    r1: T) -> [T; 3]
    where T: num_traits::Float
{
    assert!(itri < tri2vtx.len() / 3);
    let i0 = tri2vtx[itri * 3 + 0];
    let i1 = tri2vtx[itri * 3 + 1];
    let i2 = tri2vtx[itri * 3 + 2];
    let p0 = &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3];
    let p1 = &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3];
    let p2 = &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3];
    let r2 = T::one() - r0 - r1;
    [
        r0 * p0[0] + r1 * p1[0] + r2 * p2[0],
        r0 * p0[1] + r1 * p1[1] + r2 * p2[1],
        r0 * p0[2] + r1 * p1[2] + r2 * p2[2]]
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

#[allow(clippy::identity_op)]
pub fn mean_edge_length(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> f32 {
    let num_tri = tri2vtx.len() / 3;
    let mut sum = 0_f32;
    for i_tri in 0..num_tri {
        let i0 = tri2vtx[i_tri * 3 + 0];
        let i1 = tri2vtx[i_tri * 3 + 1];
        let i2 = tri2vtx[i_tri * 3 + 2];
        sum += del_geo::vec3::distance_(
            &vtx2xyz[i0 * 3..i0 * 3 + 3],
            &vtx2xyz[i1 * 3..i1 * 3 + 3]);
        sum += del_geo::vec3::distance_(
            &vtx2xyz[i1 * 3..i1 * 3 + 3],
            &vtx2xyz[i2 * 3..i2 * 3 + 3]);
        sum += del_geo::vec3::distance_(
            &vtx2xyz[i2 * 3..i2 * 3 + 3],
            &vtx2xyz[i0 * 3..i0 * 3 + 3]);
    }
    sum / (num_tri * 3) as f32
}


pub fn merge<T>(
    out_tri2vtx: &mut Vec<usize>,
    out_vtx2xyz: &mut Vec<T>,
    tri2vtx: &[usize],
    vtx2xyz: &[T])
    where T: Copy
{
    let num_vtx0 = out_vtx2xyz.len() / 3;
    tri2vtx.iter().for_each(|&v| out_tri2vtx.push(num_vtx0 + v));
    vtx2xyz.iter().for_each(|&v| out_vtx2xyz.push(v));
}


pub fn mesh_laplacian_cotangent<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    row2idx: &[usize],
    idx2col: &[usize],
    row2val: &mut [T],
    idx2val: &mut [T],
    merge_buffer: &mut Vec<usize>)
    where T: num_traits::Float + 'static + std::ops::AddAssign,
          f64: num_traits::AsPrimitive<T>
{
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let v0 = &vtx2xyz[i0 * 3..i0 * 3 + 3];
        let v1 = &vtx2xyz[i1 * 3..i1 * 3 + 3];
        let v2 = &vtx2xyz[i2 * 3..i2 * 3 + 3];
        let emat: [T; 9] = del_geo::tri3::emat_cotangent_laplacian(v0,v1,v2);
        crate::merge(
            node2vtx, node2vtx, &emat,
            row2idx, idx2col,
            row2val, idx2val,
            merge_buffer);
    }
}

pub fn optimal_rotation_for_arap<T>(
    i_vtx: usize,
    adj2vtx: &[usize],
    vtx2xyz_ini: &[T],
    vtx2xyz_def: &[T],
    adj2weight: &[T],
    weight_scale: T) -> nalgebra::Matrix3::<T>
    where T: nalgebra::RealField + Copy + std::ops::AddAssign
{
    let p0 = &vtx2xyz_ini[i_vtx * 3..i_vtx * 3 + 3];
    let p1 = &vtx2xyz_def[i_vtx * 3..i_vtx * 3 + 3];
    let mut a = nalgebra::Matrix3::<T>::zeros();
    for idx in 0..adj2vtx.len() {
        let j_vtx = adj2vtx[idx];
        let q0 = &vtx2xyz_ini[j_vtx * 3..j_vtx * 3 + 3];
        let q1 = &vtx2xyz_def[j_vtx * 3..j_vtx * 3 + 3];
        let pq0 = del_geo::vec3::sub_(q0, p0);
        let pq1 = del_geo::vec3::sub_(q1, p1);
        let w = adj2weight[idx] * weight_scale;
        a.m11 += w * pq1[0] * pq0[0];
        a.m12 += w * pq1[0] * pq0[1];
        a.m13 += w * pq1[0] * pq0[2];
        a.m21 += w * pq1[1] * pq0[0];
        a.m22 += w * pq1[1] * pq0[1];
        a.m23 += w * pq1[1] * pq0[2];
        a.m31 += w * pq1[2] * pq0[0];
        a.m32 += w * pq1[2] * pq0[1];
        a.m33 += w * pq1[2] * pq0[2];
    }
    del_geo::mat3::rotational_component(&a)
}

#[test]
fn test_optimal_rotation_for_arap() {
    let (tri2vtx, vtx2xyz_ini)
        = crate::trimesh3_primitive::capsule_yup(
        0.2, 1.6, 24, 4, 24);
    let num_vtx = vtx2xyz_ini.len() / 3;
    let (row2idx, idx2col)
        = crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx);
    let (_row2val, idx2val) = {
        let mut row2val = vec!(0f64; num_vtx);
        let mut idx2val = vec!(0f64; idx2col.len());
        let mut merge_buffer = vec!(0usize; 0);
        mesh_laplacian_cotangent(
            &tri2vtx, &vtx2xyz_ini,
            &row2idx, &idx2col,
            &mut row2val, &mut idx2val, &mut merge_buffer);
        (row2val, idx2val)
    };
    let mut vtx2xyz_def = vtx2xyz_ini.clone();
    let r0 = {
        let a_mat = nalgebra::Matrix4::<f64>::new_rotation(
            nalgebra::Vector3::<f64>::new(1., 2., 3.));
        for i_vtx in 0..vtx2xyz_def.len() / 3 {
            let p0 = nalgebra::Vector3::<f64>::new(
                vtx2xyz_ini[i_vtx*3+0],
                vtx2xyz_ini[i_vtx*3+1],
                vtx2xyz_ini[i_vtx*3+2]);
            let p1 = a_mat.transform_vector(&p0);
            vtx2xyz_def[i_vtx*3+0] = p1.x;
            vtx2xyz_def[i_vtx*3+1] = p1.y;
            vtx2xyz_def[i_vtx*3+2] = p1.z;
        }
        let r0: nalgebra::Matrix3::<f64> = a_mat.fixed_view::<3,3>(0,0).into();
        r0
    };
    for i_vtx in 0..vtx2xyz_ini.len() / 3 {
        let r = optimal_rotation_for_arap(
            i_vtx,
            &idx2col[row2idx[i_vtx]..row2idx[i_vtx+1]],
            &vtx2xyz_ini,
            &vtx2xyz_def,
            &idx2val[row2idx[i_vtx]..row2idx[i_vtx+1]], -1.);
        assert!((r.determinant()-1.0).abs()<1.0e-5);
        assert!((r-r0).norm()<1.0e-5);
    }

}