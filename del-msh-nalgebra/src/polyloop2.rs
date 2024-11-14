use num_traits::AsPrimitive;

pub fn from_circle(rad: f32, n: usize) -> nalgebra::Matrix2xX<f32> {
    let mut vtx2xy = nalgebra::Matrix2xX::<f32>::zeros(n);
    for i in 0..n {
        let theta = std::f32::consts::PI * 2_f32 * i as f32 / n as f32;
        vtx2xy.column_mut(i).x = rad * f32::cos(theta);
        vtx2xy.column_mut(i).y = rad * f32::sin(theta);
    }
    vtx2xy
}

// above: from methods
// -----------------------------------------------------

pub fn area<T>(vtx2xy: &[nalgebra::Vector2<T>]) -> T
where
    T: nalgebra::RealField + Copy,
{
    let num_vtx = vtx2xy.len();
    let zero = nalgebra::Vector2::<T>::zeros();
    let mut area = T::zero();
    for i_edge in 0..num_vtx {
        let i0 = i_edge;
        let i1 = (i_edge + 1) % num_vtx;
        let p0 = &vtx2xy[i0];
        let p1 = &vtx2xy[i1];
        area += del_geo_nalgebra::tri2::area(&zero, p0, p1);
    }
    area
}

pub fn winding_number<Real>(vtx2xy: &[nalgebra::Vector2<Real>], p: &nalgebra::Vector2<Real>) -> Real
where
    Real: num_traits::Float + num_traits::FloatConst + std::ops::AddAssign + std::fmt::Debug,
{
    let num_vtx = vtx2xy.len();
    let mut wn: Real = Real::zero();
    for i in 0..num_vtx {
        let j = (i + 1) % num_vtx;
        wn += del_geo_core::edge2::winding_number(
            &vtx2xy[i].as_slice().try_into().unwrap(),
            &vtx2xy[j].as_slice().try_into().unwrap(),
            p.as_slice().try_into().unwrap(),
        );
    }
    wn
}

pub fn distance_to_point<Real>(
    vtx2xy: &[nalgebra::Vector2<Real>],
    g: &nalgebra::Vector2<Real>,
) -> Real
where
    Real: nalgebra::RealField + Copy,
    f64: AsPrimitive<Real>,
{
    // visit all the boudnary
    let np = vtx2xy.len() / 2;
    let mut dist_min = Real::max_value().unwrap();
    for ip in 0..np {
        let jp = (ip + 1) % np;
        let pi = vtx2xy[ip];
        let pj = vtx2xy[jp];
        let dist = del_geo_nalgebra::edge::distance_to_point(g, &pi, &pj);
        if dist < dist_min {
            dist_min = dist;
        }
    }
    dist_min
}
