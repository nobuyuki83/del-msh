//! methods for 2D poly loop

use num_traits::AsPrimitive;

/// area
pub fn area<T>(
    vtx2xy: &[T]) -> T
    where T: num_traits::Float + Copy + 'static + std::ops::AddAssign,
          f64: AsPrimitive<T>
{
    let num_vtx = vtx2xy.len() / 2;
    assert_eq!(vtx2xy.len(), num_vtx * 2);
    let zero = [T::zero(), T::zero()];
    let mut area = T::zero();
    for i_edge in 0..num_vtx {
        let i0 = i_edge;
        let i1 = (i_edge + 1) % num_vtx;
        let p0 = (&vtx2xy[i0 * 2..i0 * 2 + 2]).try_into().unwrap();
        let p1 = (&vtx2xy[i1 * 2..i1 * 2 + 2]).try_into().unwrap();
        area += del_geo::tri2::area_(&zero, p0, p1);
    }
    area
}

pub fn from_circle(
    rad: f32,
    n: usize) -> nalgebra::Matrix2xX<f32> {
    let mut vtx2xy = nalgebra::Matrix2xX::<f32>::zeros(n);
    for i in 0..n {
        let theta = std::f32::consts::PI * 2_f32 * i as f32 / n as f32;
        vtx2xy.column_mut(i).x = rad * f32::cos(theta);
        vtx2xy.column_mut(i).y = rad * f32::sin(theta);
    }
    vtx2xy
}

pub fn from_pentagram<Real>(
    center: &[Real],
    scale: Real) -> Vec<Real>
    where Real: num_traits::Float + 'static + Copy,
          f64: AsPrimitive<Real>,
          usize: AsPrimitive<Real>
{
    let dt: Real = (std::f64::consts::PI / 5_f64).as_();
    let hp: Real = (std::f64::consts::FRAC_PI_2).as_();
    let ratio = (2f64 / (3f64 + 5f64.sqrt())).as_();
    let mut xys = Vec::<Real>::new();
    for i in 0..10usize {
        let rad = if i % 2 == 0 { scale } else { ratio * scale };
        xys.push((dt * i.as_()+hp).cos() * rad + center[0]);
        xys.push((dt * i.as_()+hp).sin() * rad + center[1]);
    }
    xys
}

pub fn is_inside<Real>(
    vtx2xy: &[Real],
    p: &[Real;2]) -> bool
    where Real: num_traits::Float + Copy + 'static + std::ops::AddAssign,
          f64: AsPrimitive<Real>
{
    let num_vtx = vtx2xy.len() / 2;
    let mut wn: Real = Real::zero();
    for i in 0..num_vtx {
        let j = (i + 1) % num_vtx;
        wn += del_geo::edge2::winding_number_(
            (&vtx2xy[i * 2..(i + 1) * 2]).try_into().unwrap(),
            (&vtx2xy[j * 2..(j + 1) * 2]).try_into().unwrap(),
            p);
    }
    if (wn - Real::one()).abs() < 0.1.as_() { return true; }
    false
}

pub fn distance_to_point<Real>(
    vtx2xy: &[Real],
    p: &[Real]) -> Real
where Real: nalgebra::RealField + Copy,
    f64: AsPrimitive<Real>
{
    let g = nalgebra::Vector2::<Real>::from_row_slice(p);
    // visit all the boudnary
    let np = vtx2xy.len() / 2;
    let mut dist_min = Real::max_value().unwrap();
    for ip in 0..np {
        let jp = (ip + 1) % np;
        let pi = del_geo::vec2::to_na(vtx2xy, ip);
        let pj = del_geo::vec2::to_na(vtx2xy, jp);
        let dist = del_geo::edge::distance_to_point(&g, &pi, &pj);
        if dist < dist_min {
            dist_min = dist;
        }
    }
    dist_min
}

pub fn to_uniform_density_random_points<Real>(
    vtx2xy: &[Real],
    cell_len: Real,
    rng: &mut rand::rngs::StdRng) -> Vec<Real>
    where Real: num_traits::Float + std::ops::AddAssign + Copy + 'static + AsPrimitive<usize>,
          rand::distributions::Standard: rand::prelude::Distribution<Real>,
          f64: AsPrimitive<Real>,
          usize: AsPrimitive<Real>
{
    let aabb = del_geo::aabb2::from_vtx2xy(vtx2xy);
    use rand::Rng;
    let base_pos = [aabb[0] - cell_len * rng.gen::<Real>(), aabb[1] - cell_len * rng.gen::<Real>()];
    let nx = ((aabb[2] - base_pos[0]) / cell_len).as_() + 1;
    let ny = ((aabb[3] - base_pos[1]) / cell_len).as_() + 1;
    let mut res = vec!();
    for ix in 0..nx {
        for iy in 0..ny {
            let x = base_pos[0] + (ix.as_() + rng.gen::<Real>()) * cell_len;
            let y = base_pos[1] + (iy.as_() + rng.gen::<Real>()) * cell_len;
            let is_inside = is_inside(vtx2xy, &[x, y]);
            if !is_inside { continue; }
            res.push(x);
            res.push(y);
        }
    }
    res
}

#[allow(clippy::identity_op)]
pub fn to_svg<Real>(
    vtx2xy: &[Real],
    transform: &nalgebra::Matrix3<Real>) -> String
where Real: std::fmt::Display + Copy + nalgebra::RealField
{
    let mut res = String::new();
    for ivtx in 0..vtx2xy.len() / 2 {
        let x = vtx2xy[ivtx*2+0];
        let y = vtx2xy[ivtx*2+1];
        let a = transform * nalgebra::Vector3::<Real>::new(x, y, Real::one());
        res += format!("{} {}", a.x, a.y).as_str();
        if ivtx != vtx2xy.len()/2 - 1 {
            res += ",";
        }
    }
    res
}

// -----------------------------------------------------

#[test]
fn test_circle() {
    let vtx2xy0 = from_circle(1.0, 300);
    let arclen0 = crate::polyloop::arclength::<f32, 2>(vtx2xy0.as_slice());
    assert!((arclen0 - 2. * std::f32::consts::PI).abs() < 1.0e-3);
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
        assert!((min_edge_len1 - arclen1 / ndiv1 as f32).abs() < 1.0e-3);
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
        assert!((min_edge_len2 - arclen2 / ndiv2 as f32).abs() < 1.0e-3);
    }
}