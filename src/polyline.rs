//! methods for polyline mesh

use num_traits::AsPrimitive;

/// resample the input polyline with a fix interval
/// the first point of the input point will be preserved
/// the output polyline will be shorter than the input
pub fn resample<T, const NDIM: usize>(
    stroke0: &Vec<nalgebra::base::SVector<T, NDIM>>,
    l: T) -> Vec<nalgebra::base::SVector<T, NDIM>>
    where T: nalgebra::RealField + Copy,
          f64: num_traits::AsPrimitive<T>
{
    if stroke0.is_empty() {
        return vec!();
    }
    let mut stroke = Vec::<nalgebra::base::SVector<T, NDIM>>::new();
    stroke.push(stroke0[0]);
    let mut jcur = 0;
    let mut rcur: T = 0_f64.as_();
    let mut lcur = l;
    loop {
        if jcur >= stroke0.len() - 1 { break; }
        let lenj = (stroke0[jcur + 1] - stroke0[jcur]).norm();
        let lenjr = lenj * (1_f64.as_() - rcur);
        if lenjr > lcur { // put point in this segment
            rcur += lcur / lenj;
            stroke.push(stroke0[jcur].scale(1_f64.as_() - rcur) + stroke0[jcur + 1].scale(rcur));
            lcur = l;
        } else { // next segment
            lcur -= lenjr;
            rcur = 0_f64.as_();
            jcur += 1;
        }
    }
    stroke
}

pub fn resample_preserve_corner<T, const NDIM: usize>(
    stroke0: &Vec<nalgebra::base::SVector<T, NDIM>>,
    l: T) -> Vec<nalgebra::base::SVector<T, NDIM>>
    where T: nalgebra::RealField + Copy + AsPrimitive<usize>,
          f64: num_traits::AsPrimitive<T>,
          usize: AsPrimitive<T>
{
    if stroke0.is_empty() {
        return vec!();
    }
    let mut stroke = Vec::<nalgebra::base::SVector<T, NDIM>>::new();
    let num_pnt0 = stroke0.len();
    for i_seg0 in 0..num_pnt0 - 1 {
        let p0 = stroke0[i_seg0];
        let q0 = stroke0[i_seg0+1];
        let len = (p0-q0).norm();
        let np_new: usize = (len / l).as_();
        let dr = T::one() / (np_new+1).as_();
        stroke.push(stroke0[i_seg0]);
        for ip_new in 1..np_new+1 {
            let p_new = p0 + (q0-p0).scale(ip_new.as_()*dr);
            stroke.push(p_new);
        }
    }
    stroke.push(stroke0[stroke0.len()-1]);
    stroke
}