//! methods for 3D triangle mesh

pub fn normal(
    tri2vtx: &[usize],
    vtx2xyz: &[f64]) -> Vec<f64> {
    let mut vtx2nrm = vec!(0_f64; vtx2xyz.len());
    for t in tri2vtx.chunks(3) {
        let p0 = &vtx2xyz[t[0] * 3..t[0] * 3 + 3];
        let p1 = &vtx2xyz[t[1] * 3..t[1] * 3 + 3];
        let p2 = &vtx2xyz[t[2] * 3..t[2] * 3 + 3];
        let mut un = [0_f64; 3];
        del_geo::tri3::unit_normal_(&mut un, p0, p1, p2);
        for i in 0..3 {
            vtx2nrm[t[i] * 3 + 0] += un[0];
            vtx2nrm[t[i] * 3 + 1] += un[1];
            vtx2nrm[t[i] * 3 + 2] += un[2];
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

pub fn areas(
    tri2vtx: &[usize],
    vtx2xyz: &[f32], ) -> Vec<f32>
{
    let mut tri2area = vec!();
    tri2area.reserve(tri2vtx.len() / 3);
    for idx_tri in tri2vtx.chunks(3) {
        let i0 = idx_tri[0];
        let i1 = idx_tri[1];
        let i2 = idx_tri[2];
        let area = del_geo::tri3::area_(
            &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3],
            &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3],
            &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3]);
        tri2area.push(area);
    }
    tri2area
}

pub fn area_par_vertex(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> Vec<f32> {
    let num_vtx = vtx2xyz.len() / 3;
    let num_tri = tri2vtx.len() / 3;
    let mut areas = vec!(0_f32; num_vtx);
    for i_tri in 0..num_tri {
        let i0 = tri2vtx[i_tri * 3 + 0];
        let i1 = tri2vtx[i_tri * 3 + 1];
        let i2 = tri2vtx[i_tri * 3 + 2];
        let a0 = del_geo::tri3::area_(
            &vtx2xyz[i0 * 3..i0 * 3 + 3],
            &vtx2xyz[i1 * 3..i1 * 3 + 3],
            &vtx2xyz[i2 * 3..i2 * 3 + 3]) / 3_f32;
        areas[i0] += a0;
        areas[i1] += a0;
        areas[i2] += a0;
    }
    areas
}

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


pub fn merge(
    out_tri2vtx: &mut Vec<usize>,
    out_vtx2xyz: &mut Vec<f64>,
    tri2vtx: &[usize],
    vtx2xyz: &[f64])
{
    let num_vtx0 = out_vtx2xyz.len() / 3;
    tri2vtx.iter().for_each(|&v| out_tri2vtx.push(num_vtx0+v));
    vtx2xyz.iter().for_each(|&v| out_vtx2xyz.push(v));
}