//! methods for polyline mesh

use num_traits::AsPrimitive;

pub fn cov<T, const N: usize>(vtx2xyz: &[T]) -> [[T; N]; N]
where
    T: num_traits::Float + Copy + 'static + std::iter::Sum,
    f64: AsPrimitive<T>,
{
    let one = T::one();
    let three = one + one + one;
    let six = three + three;
    let num_vtx = vtx2xyz.len() / N;
    assert_eq!(vtx2xyz.len(), num_vtx * N);
    let cog: [T; N] = crate::polyloop::cog_as_edges::<T, N>(vtx2xyz);
    let mut cov = [[T::zero(); N]; N];
    for i_edge in 0..num_vtx - 1 {
        let iv0 = i_edge;
        let iv1 = i_edge + 1;
        use del_geo_core::vecn::VecN;
        let q0: &[T; N] = &vtx2xyz[iv0 * N..iv0 * N + N].try_into().unwrap();
        let q0 = q0.sub(&cog);
        let q1: &[T; N] = &vtx2xyz[iv1 * N..iv1 * N + N].try_into().unwrap();
        let q1 = q1.sub(&cog);
        let l = q0.sub(&q1).norm();
        for i in 0..N {
            for j in 0..N {
                cov[i][j] = cov[i][j]
                    + (q0[i] * q0[j] + q1[i] * q1[j]) * (l / three)
                    + (q0[i] * q1[j] + q1[i] * q0[j]) * (l / six);
            }
        }
    }
    cov
}

/// resample the input polyline with a fix interval
/// the first point of the input point will be preserved
/// the output polyline will be shorter than the input
pub fn resample<T, const NDIM: usize>(stroke0: &[[T; NDIM]; NDIM], l: T) -> Vec<[T; NDIM]>
where
    T: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<T>,
{
    use del_geo_core::vecn::VecN;
    if stroke0.is_empty() {
        return vec![];
    }
    let mut stroke = Vec::<[T; NDIM]>::new();
    stroke.push(stroke0[0]);
    let mut jcur = 0;
    let mut rcur: T = 0_f64.as_();
    let mut lcur = l;
    loop {
        if jcur >= stroke0.len() - 1 {
            break;
        }
        let lenj = del_geo_core::vecn::distance(&stroke0[jcur + 1], &stroke0[jcur]);
        let lenjr = lenj * (1_f64.as_() - rcur);
        if lenjr > lcur {
            // put point in this segment
            rcur = rcur + lcur / lenj;
            let p = stroke0[jcur]
                .scale(1_f64.as_() - rcur)
                .add(&stroke0[jcur + 1].scale(rcur));
            stroke.push(p);
            lcur = l;
        } else {
            // next segment
            lcur = lcur - lenjr;
            rcur = 0_f64.as_();
            jcur += 1;
        }
    }
    stroke
}

pub fn resample_preserve_corner<T, const NDIM: usize>(stroke0: &[[T; NDIM]], l: T) -> Vec<[T; NDIM]>
where
    T: num_traits::Float + Copy + AsPrimitive<usize>,
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    use del_geo_core::vecn::VecN;
    if stroke0.is_empty() {
        return vec![];
    }
    let mut stroke = Vec::<[T; NDIM]>::new();
    let num_pnt0 = stroke0.len();
    for i_seg0 in 0..num_pnt0 - 1 {
        let p0 = stroke0[i_seg0];
        let q0 = stroke0[i_seg0 + 1];
        let len = del_geo_core::vecn::distance(&p0, &q0);
        let np_new: usize = (len / l).as_();
        let dr = T::one() / (np_new + 1).as_();
        stroke.push(stroke0[i_seg0]);
        for ip_new in 1..np_new + 1 {
            let p_new = p0.add(&q0.sub(&p0).scale(ip_new.as_() * dr));
            stroke.push(p_new);
        }
    }
    stroke.push(stroke0[stroke0.len() - 1]);
    stroke
}
