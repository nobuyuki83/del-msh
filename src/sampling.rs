//! stochastic sampling on mesh

use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn cumulative_areas_trimesh3_condition<F: Fn(usize) -> bool, T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T],
    tri2isvalid: F) -> Vec<T>
where T: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<T>
{
    let num_tri = tri2vtx.len() / 3;
    assert_eq!(tri2vtx.len(), num_tri * 3);
    let mut cumulative_area_sum = Vec::<T>::with_capacity(num_tri + 1);
    cumulative_area_sum.push(0_f64.as_());
    for idx_tri in 0..num_tri {
        let a0 = if !tri2isvalid(idx_tri) {
            0_f64.as_()
        } else {
            let i0 = tri2vtx[idx_tri * 3 + 0];
            let i1 = tri2vtx[idx_tri * 3 + 1];
            let i2 = tri2vtx[idx_tri * 3 + 2];
            del_geo::tri3::area_(
                &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3],
                &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3],
                &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3])
        };
        let t0 = cumulative_area_sum[cumulative_area_sum.len() - 1];
        cumulative_area_sum.push(a0 + t0);
    }
    cumulative_area_sum
}


pub fn cumulative_area_sum<T>(
    tri2vtx: &[usize],
    vtx2xyz: &[T]) -> Vec<T>
    where T: num_traits::Float + Copy + 'static,
          f64: AsPrimitive<T>
{
    cumulative_areas_trimesh3_condition(
        tri2vtx, vtx2xyz, |_itri| { true })
}

pub fn sample_uniformly_trimesh<T>(
    cumulative_area_sum: &[T],
    val01: T,
    r1: T) -> (usize, T, T)
where T: num_traits::Float
{
    let ntri = cumulative_area_sum.len() - 1;
    let a0 = val01 * cumulative_area_sum[ntri];
    let mut itri_l = 0;
    let mut itri_u = ntri;
    loop {  // bisection method
        assert!(cumulative_area_sum[itri_l] < a0);
        assert!(a0 <= cumulative_area_sum[itri_u]);
        let itri_h = (itri_u + itri_l) / 2;
        if itri_u - itri_l == 1 { break; }
        if cumulative_area_sum[itri_h] < a0 {
            itri_l = itri_h;
        } else {
            itri_u = itri_h;
        }
    }
    assert!(cumulative_area_sum[itri_l] < a0);
    assert!(a0 <= cumulative_area_sum[itri_l + 1]);
    let r0 = (a0 - cumulative_area_sum[itri_l]) / (cumulative_area_sum[itri_l + 1] - cumulative_area_sum[itri_l]);
    if r0 + r1 > T::one() {
        let r0a = r0;
        let r1a = r1;
        return (itri_l, T::one() - r1a, T::one() - r0a);
    }
    (itri_l, r0, r1)
}

#[allow(clippy::identity_op)]
pub fn position_on_trimesh3<T>(
    itri: usize,
    r0: T,
    r1: T,
    tri2vtx: &[usize],
    vtx2xyz: &[T]) -> [T; 3]
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

