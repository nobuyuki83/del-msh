pub fn position_from_barycentric_coordinate<Real, const N: usize>(
    tri2vtx: &[usize],
    vtx2xyz: &[Real],
    i_tri: usize,
    r0: Real,
    r1: Real,
) -> [Real; N]
where
    Real: num_traits::Float,
{
    assert!(i_tri < tri2vtx.len() / 3);
    let i0 = tri2vtx[i_tri * 3];
    let i1 = tri2vtx[i_tri * 3 + 1];
    let i2 = tri2vtx[i_tri * 3 + 2];
    let p0 = &vtx2xyz[i0 * N..i0 * N + N];
    let p1 = &vtx2xyz[i1 * N..i1 * N + N];
    let p2 = &vtx2xyz[i2 * N..i2 * N + N];
    let r2 = Real::one() - r0 - r1;
    let mut res = [Real::zero(); N];
    for i in 0..N {
        res[i] = r0 * p0[i] + r1 * p1[i] + r2 * p2[i];
    }
    res
}

pub fn tri2cumsumarea_with_condition<F: Fn(usize) -> bool, Real>(
    tri2vtx: &[usize],
    vtx2xyz: &[Real],
    num_dim: usize,
    tri2isvalid: F,
) -> Vec<Real>
where
    Real: num_traits::Float + std::fmt::Debug + std::ops::MulAssign,
{
    assert!(num_dim == 2 || num_dim == 3);
    let num_tri = tri2vtx.len() / 3;
    assert_eq!(tri2vtx.len(), num_tri * 3);
    let mut cumulative_area_sum = Vec::<Real>::with_capacity(num_tri + 1);
    cumulative_area_sum.push(Real::zero());
    for idx_tri in 0..num_tri {
        let a0 = if !tri2isvalid(idx_tri) {
            Real::zero()
        } else if num_dim == 2 {
            crate::trimesh2::to_tri2(idx_tri, tri2vtx, vtx2xyz).area()
        } else {
            crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, idx_tri).area()
        };
        let t0 = cumulative_area_sum[cumulative_area_sum.len() - 1];
        cumulative_area_sum.push(a0 + t0);
    }
    cumulative_area_sum
}

pub fn tri2cumsumarea<Real>(tri2vtx: &[usize], vtx2xyz: &[Real], num_dim: usize) -> Vec<Real>
where
    Real: num_traits::Float + std::fmt::Debug + std::ops::MulAssign,
{
    tri2cumsumarea_with_condition(tri2vtx, vtx2xyz, num_dim, |_itri| true)
}

/// sample points uniformly inside triangle mesh
/// * val01_a - uniformly sampled float value `[0,1]`
/// * val01_b - uniformly sampled float value `[0,1]`
/// # Return
/// (i_tri: usize, r0: Real, r1: Real)
pub fn sample_uniformly<Real>(
    tri2cumsumarea: &[Real],
    val01_a: Real,
    val01_b: Real,
) -> (usize, Real, Real)
where
    Real: num_traits::Float + std::fmt::Debug,
{
    let (i_tri_l, r0, _p0) = crate::cumsum::sample(tri2cumsumarea, val01_a);
    if r0 + val01_b > Real::one() {
        let r0a = r0;
        let r1a = val01_b;
        return (i_tri_l, Real::one() - r1a, Real::one() - r0a);
    }
    (i_tri_l, r0, val01_b)
}
