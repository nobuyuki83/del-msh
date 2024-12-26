/// # Return
/// - (i: usize, r1: Real, p0: Real)
///   * `i` - selected element
///   * `r1` - uniform distribution inside selected element
///   * `p0` - probability that selected element is sampled
pub fn sample<Real>(cumulative_area_sum: &[Real], val01_a: Real) -> (usize, Real, Real)
where
    Real: num_traits::Float + std::fmt::Debug,
{
    let num_tri = cumulative_area_sum.len() - 1;
    let a0: Real = val01_a * cumulative_area_sum[num_tri];
    let mut i_tri_l = 0;
    let mut i_tri_u = num_tri;
    loop {
        // bisection method
        assert!(
            cumulative_area_sum[i_tri_l] <= a0,
            "{:?} {:?} {:?}",
            i_tri_l,
            cumulative_area_sum[i_tri_l],
            a0
        );
        assert!(a0 <= cumulative_area_sum[i_tri_u]);
        let i_tri_h = (i_tri_u + i_tri_l) / 2;
        if i_tri_u - i_tri_l == 1 {
            break;
        }
        if cumulative_area_sum[i_tri_h] < a0 {
            i_tri_l = i_tri_h;
        } else {
            i_tri_u = i_tri_h;
        }
    }
    let d0 = cumulative_area_sum[i_tri_l + 1] - cumulative_area_sum[i_tri_l];
    let r0 = (a0 - cumulative_area_sum[i_tri_l]) / d0;
    let p0 = d0 / cumulative_area_sum[num_tri];
    assert!(cumulative_area_sum[i_tri_l] <= a0);
    assert!(a0 <= cumulative_area_sum[i_tri_l + 1]);
    (i_tri_l, r0, p0)
}
