//! methods for polyline mesh

use num_traits::AsPrimitive;

pub fn resample<T, const X: usize>(
    stroke0: &Vec<nalgebra::base::SVector<T, X>>,
    l: T) -> Vec<nalgebra::base::SVector<T, X>>
    where T: nalgebra::RealField + Copy,
          f64: num_traits::AsPrimitive<T>
{
    if stroke0.is_empty() {
        return vec!();
    }
    let mut stroke = Vec::<nalgebra::base::SVector<T, X>>::new();
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

pub fn parallel_transport_polyline<T>(
    vtx2xyz: &[T]) -> nalgebra::Matrix3xX::<T>
where T: nalgebra::RealField + 'static + Copy,
    f64: num_traits::AsPrimitive<T>
{
    use del_geo::vec3::navec3;
    let num_vtx = vtx2xyz.len() / 3;
    let mut vtx2bin = nalgebra::Matrix3xX::<T>::zeros(num_vtx);
    {   // first segment
        let v01 = (navec3(vtx2xyz, 1) - navec3(vtx2xyz,0)).into_owned();
        let (x, _) = del_geo::vec3::frame_from_z_vector(v01);
        vtx2bin.column_mut(0).copy_from(&x);
    }
    for iseg1 in 1..num_vtx {
        let iv0 = iseg1 - 1;
        let iv1 = iseg1;
        let iv2 = (iseg1 + 1) % num_vtx;
        let iseg0 = iseg1 - 1;
        let v01 = navec3(vtx2xyz,iv1) - navec3(vtx2xyz,iv0);
        let v12 = navec3(vtx2xyz,iv2) - navec3(vtx2xyz,iv1);
        let rot = del_geo::mat3::minimum_rotation_matrix(v01, v12);
        let b01: nalgebra::Vector3::<T> = vtx2bin.column(iseg0).into_owned();
        let b12: nalgebra::Vector3::<T> = rot * b01;
        vtx2bin.column_mut(iseg1).copy_from(&b12);
    }
    vtx2bin
}