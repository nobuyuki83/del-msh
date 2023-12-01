//! methods for 2D poly loop

use num_traits::AsPrimitive;


/// area
pub fn area<T>(
    vtx2xy: &[T]) -> T
where T: num_traits::Float + Copy + 'static + std::ops::AddAssign,
      f64: AsPrimitive<T>
{
    let num_vtx = vtx2xy.len() / 2;
    assert_eq!(vtx2xy.len(), num_vtx*2);
    let zero = [T::zero(), T::zero()];
    let mut area = T::zero();
    for i_edge in 0..num_vtx {
        let i0 = i_edge;
        let i1 = (i_edge + 1) % num_vtx;
        let p0 = &vtx2xy[i0*2..i0*2+2];
        let p1 = &vtx2xy[i1*2..i1*2+2];
        area += del_geo::tri2::area_(&zero, p0,p1);
    }
    area
}

pub fn from_circle(
    rad: f32,
    n: usize) -> nalgebra::Matrix2xX::<f32>{
    let mut vtx2xy = nalgebra::Matrix2xX::<f32>::zeros(n);
    for i in 0..n {
        let theta = std::f32::consts::PI * 2_f32 * i as f32 / n as f32;
        vtx2xy.column_mut(i).x = rad * f32::cos(theta);
        vtx2xy.column_mut(i).y = rad * f32::sin(theta);
    }
    vtx2xy
}

#[test]
fn test_circle() {
    let vtx2xy0 = from_circle(1.0, 300);
    let arclen0 = crate::polyloop::arclength::<f32,2>(vtx2xy0.as_slice());
    assert!((arclen0-2.*std::f32::consts::PI).abs()<1.0e-3);
    //
    {
        let ndiv1 = 330;
        let vtx2xy1 = crate::polyloop::resample::<f32, 2>(vtx2xy0.as_slice(), ndiv1);
        assert_eq!(vtx2xy1.len(), ndiv1 * 2);
        let arclen1 = crate::polyloop::arclength::<f32, 2>(vtx2xy1.as_slice());
        assert!((arclen0 - arclen1).abs() < 1.0e-3);
        let edge2length1 = crate::polyloop::edge2length::<f32, 2>(vtx2xy1.as_slice());
        let min_edge_len1 = edge2length1.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        assert!((min_edge_len1-arclen1/ndiv1 as f32).abs()<1.0e-3);
    }
    {
        let ndiv2 = 156;
        let vtx2xy2 = crate::polyloop::resample::<f32, 2>(vtx2xy0.as_slice(), ndiv2);
        assert_eq!(vtx2xy2.len(), ndiv2 * 2);
        let arclen2 = crate::polyloop::arclength::<f32, 2>(vtx2xy2.as_slice());
        assert!((arclen0 - arclen2).abs() < 1.0e-3);
        let edge2length2 = crate::polyloop::edge2length::<f32, 2>(vtx2xy2.as_slice());
        let min_edge_len2 = edge2length2.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        assert!((min_edge_len2-arclen2/ndiv2 as f32).abs()<1.0e-3);
    }
}