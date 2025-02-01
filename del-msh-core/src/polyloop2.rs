//! methods for 2D poly loop

use num_traits::AsPrimitive;

pub fn winding_number_<Real>(vtx2xy: &[Real], p: &[Real; 2]) -> Real
where
    Real: num_traits::Float + num_traits::FloatConst + std::ops::AddAssign,
{
    let num_vtx = vtx2xy.len() / 2;
    let mut wn: Real = Real::zero();
    for i in 0..num_vtx {
        let j = (i + 1) % num_vtx;
        wn += del_geo_core::edge2::winding_number(
            arrayref::array_ref![vtx2xy, i * 2, 2],
            arrayref::array_ref![vtx2xy, j * 2, 2],
            p,
        );
    }
    wn
}

pub fn is_inside_<Real>(vtx2xy: &[Real], p: &[Real; 2]) -> bool
where
    Real: num_traits::Float + num_traits::FloatConst + std::ops::AddAssign,
{
    let wn = winding_number_(vtx2xy, p);
    let one = Real::one();
    let thres = one / (one + one + one + one + one);
    if (wn - one).abs() < thres {
        return true;
    }
    false
}

/// area
pub fn area_<T>(vtx2xy: &[T]) -> T
where
    T: num_traits::Float + Copy + 'static + std::ops::AddAssign,
    f64: AsPrimitive<T>,
{
    let num_vtx = vtx2xy.len() / 2;
    assert_eq!(vtx2xy.len(), num_vtx * 2);
    let zero = [T::zero(), T::zero()];
    let mut area = T::zero();
    for i_edge in 0..num_vtx {
        let i0 = i_edge;
        let i1 = (i_edge + 1) % num_vtx;
        let p0 = arrayref::array_ref![vtx2xy, i0 * 2, 2];
        let p1 = arrayref::array_ref![vtx2xy, i1 * 2, 2];
        area += del_geo_core::tri2::area(&zero, p0, p1);
    }
    area
}

/// center of the gravity of a area bounded by this polyloop
pub fn cog_as_face<T>(vtx2xy: &[T]) -> [T; 2]
where
    T: num_traits::Float + std::ops::AddAssign + std::ops::DivAssign,
{
    let frac_three = T::one() / (T::one() + T::one() + T::one());
    let num_vtx = vtx2xy.len() / 2;
    assert_eq!(vtx2xy.len(), num_vtx * 2);
    let zero = [T::zero(); 2];
    let mut area = T::zero();
    let mut cog = [T::zero(); 2];
    for i_edge in 0..num_vtx {
        let i0 = i_edge;
        let i1 = (i_edge + 1) % num_vtx;
        let p0 = arrayref::array_ref![vtx2xy, i0 * 2, 2];
        let p1 = arrayref::array_ref![vtx2xy, i1 * 2, 2];
        let area0 = del_geo_core::tri2::area(&zero, p0, p1);
        area += area0;
        cog[0] += (p0[0] + p1[0]) * frac_three * area0;
        cog[1] += (p0[1] + p1[1]) * frac_three * area0;
    }
    cog[0] /= area;
    cog[1] /= area;
    cog
}

#[test]
fn test_cog_() {
    let vtx2xy: Vec<f32> = vec![
        -1.0, -5.0, -0.5, -5.0, 0.5, -5.0, 1.0, -5.0, 1.0, 5.0, -1.0, 5.0,
    ];
    let cog = cog_as_face(&vtx2xy);
    assert!(cog[0].abs() < 1.0e-8);
    assert!(cog[1].abs() < 1.0e-8);
}

/// star shape
pub fn from_pentagram<Real>(center: &[Real], scale: Real) -> Vec<Real>
where
    Real: num_traits::Float + 'static + Copy,
    f64: AsPrimitive<Real>,
    usize: AsPrimitive<Real>,
{
    let dt: Real = (std::f64::consts::PI / 5_f64).as_();
    let hp: Real = (std::f64::consts::FRAC_PI_2).as_();
    let ratio = (2f64 / (3f64 + 5f64.sqrt())).as_();
    let mut xys = Vec::<Real>::new();
    for i in 0..10usize {
        let rad = if i % 2 == 0 { scale } else { ratio * scale };
        xys.push((dt * i.as_() + hp).cos() * rad + center[0]);
        xys.push((dt * i.as_() + hp).sin() * rad + center[1]);
    }
    xys
}

pub fn from_circle(rad: f32, n: usize) -> Vec<f32> {
    let mut vtx2xy = vec![0f32; 2 * n];
    for i in 0..n {
        let theta = std::f32::consts::PI * 2_f32 * i as f32 / n as f32;
        vtx2xy[i * 2] = rad * f32::cos(theta);
        vtx2xy[i * 2 + 1] = rad * f32::sin(theta);
    }
    vtx2xy
}

pub fn distance_to_point_<Real>(vtx2xy: &[Real], g: &[Real; 2]) -> Real
where
    Real: num_traits::Float + Copy + 'static,
    f64: AsPrimitive<Real>,
{
    // visit all the boudnary
    let np = vtx2xy.len() / 2;
    let mut dist_min = Real::max_value();
    for ip in 0..np {
        let jp = (ip + 1) % np;
        let pi = crate::vtx2xy::to_vec2(vtx2xy, ip);
        let pj = crate::vtx2xy::to_vec2(vtx2xy, jp);
        let (dist, _p_near) = del_geo_core::edge2::nearest_point2(pi, pj, g);
        if dist < dist_min {
            dist_min = dist;
        }
    }
    dist_min
}

pub fn moment_of_inertia(vtx2xy: &[f32], pivot: &[f32; 2]) -> f32 {
    use del_geo_core::vec2;
    let ne = vtx2xy.len() / 2;
    let mut sum_i = 0.0;
    for ie in 0..ne {
        let ip0 = ie;
        let ip1 = (ie + 1) % ne;
        let p0 = [vtx2xy[ip0 * 2] - pivot[0], vtx2xy[ip0 * 2 + 1] - pivot[1]];
        let p1 = [vtx2xy[ip1 * 2] - pivot[0], vtx2xy[ip1 * 2 + 1] - pivot[1]];
        let a0 = vec2::area_quadrilateral(&p0, &p1) * 0.5;
        sum_i += a0 * (vec2::dot(&p0, &p0) + vec2::dot(&p0, &p1) + vec2::dot(&p1, &p1));
    }
    sum_i * (1.0 / 6.0)
}

/// signed distance function
/// * `vtx2xy` - flat array of coordinates
/// * `q` - pont to be evaluated
pub fn wdw_sdf_(vtx2xy: &[f32], q: &[f32; 2]) -> (f32, [f32; 2]) {
    use del_geo_core::vec2;
    let nej = vtx2xy.len() / 2;
    let mut min_dist = -1.0;
    let mut winding_number = 0f32;
    let mut pos_near = [0f32; 2];
    let mut ie_near = 0;
    for iej in 0..nej {
        let ps = arrayref::array_ref!(vtx2xy, (iej % nej) * 2, 2);
        let pe = arrayref::array_ref!(vtx2xy, ((iej + 1) % nej) * 2, 2);
        winding_number += del_geo_core::edge2::winding_number(ps, pe, q);
        let (_rm, pm) = del_geo_core::edge2::nearest_point2(ps, pe, q);
        let dist0 = del_geo_core::edge2::length(&pm, q);
        if min_dist > 0. && dist0 > min_dist {
            continue;
        }
        min_dist = dist0;
        pos_near = pm;
        ie_near = iej;
    }
    //
    let normal_out = {
        // if distance is small use edge's normal
        let ps = arrayref::array_ref!(vtx2xy, (ie_near % nej) * 2, 2);
        let pe = arrayref::array_ref!(vtx2xy, ((ie_near + 1) % nej) * 2, 2);
        let ne = vec2::sub(pe, ps);
        let ne = vec2::rotate(&ne, -std::f32::consts::PI * 0.5);
        vec2::normalize(&ne)
    };
    //
    // dbg!(winding_number);
    if (winding_number - 1.0).abs() < 0.5 {
        // inside
        let normal = if min_dist < 1.0e-5 {
            normal_out
        } else {
            vec2::normalize(&vec2::sub(&pos_near, q))
        };
        (-min_dist, normal)
    } else {
        let normal = if min_dist < 1.0e-5 {
            normal_out
        } else {
            vec2::normalize(&vec2::sub(q, &pos_near))
        };
        (min_dist, normal)
    }
}

#[test]
fn test_polygon2_sdf() {
    let vtx2xy = vec![0., 0., 1.0, 0.0, 1.0, 0.2, 0.0, 0.2];
    use del_geo_core::vec2;
    {
        let (sdf, normal) = wdw_sdf_(&vtx2xy, &[0.01, 0.1]);
        assert!((sdf + 0.01).abs() < 1.0e-5);
        assert!(vec2::length(&vec2::sub(&normal, &[-1., 0.])) < 1.0e-5);
    }
    {
        let (sdf, normal) = wdw_sdf_(&vtx2xy, &[-0.01, 0.1]);
        assert!((sdf - 0.01).abs() < 1.0e-5);
        assert!(vec2::length(&vec2::sub(&normal, &[-1., 0.])) < 1.0e-5);
    }
}

pub fn to_uniform_density_random_points<Real>(
    vtx2xy: &[Real],
    cell_len: Real,
    rng: &mut rand::rngs::StdRng,
) -> Vec<Real>
where
    Real: num_traits::Float + num_traits::FloatConst + std::ops::AddAssign + AsPrimitive<usize>,
    rand::distr::StandardUniform: rand::distr::Distribution<Real>,
    usize: AsPrimitive<Real>,
{
    let aabb = crate::vtx2xy::aabb2(vtx2xy);
    use rand::Rng;
    let base_pos = [
        aabb[0] - cell_len * rng.random::<Real>(),
        aabb[1] - cell_len * rng.random::<Real>(),
    ];
    let nx = ((aabb[2] - base_pos[0]) / cell_len).as_() + 1;
    let ny = ((aabb[3] - base_pos[1]) / cell_len).as_() + 1;
    let mut res = vec![];
    for ix in 0..nx {
        for iy in 0..ny {
            let x = base_pos[0] + (ix.as_() + rng.random::<Real>()) * cell_len;
            let y = base_pos[1] + (iy.as_() + rng.random::<Real>()) * cell_len;
            let is_inside = is_inside_(vtx2xy, &[x, y]);
            if !is_inside {
                continue;
            }
            res.push(x);
            res.push(y);
        }
    }
    res
}

#[allow(clippy::identity_op)]
pub fn to_svg<Real>(vtx2xy: &[Real], transform: &[Real; 9]) -> String
where
    Real: std::fmt::Display + Copy + num_traits::Float,
{
    let mut res = String::new();
    for ivtx in 0..vtx2xy.len() / 2 {
        let x = vtx2xy[ivtx * 2 + 0];
        let y = vtx2xy[ivtx * 2 + 1];
        let a = del_geo_core::mat3_col_major::transform_homogeneous(transform, &[x, y]).unwrap();
        res += format!("{} {}", a[0], a[1]).as_str();
        if ivtx != vtx2xy.len() / 2 - 1 {
            res += ",";
        }
    }
    res
}

#[test]
fn test_circle() {
    let vtx2xy0 = del_msh_nalgebra::polyloop2::from_circle(1.0, 300);
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
        let min_edge_len1 = edge2length1
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!((min_edge_len1 - arclen1 / ndiv1 as f32).abs() < 1.0e-3);
    }
    {
        let ndiv2 = 156;
        let vtx2xy2 = crate::polyloop::resample::<f32, 2>(vtx2xy0.as_slice(), ndiv2);
        assert_eq!(vtx2xy2.len(), ndiv2 * 2);
        let arclen2 = crate::polyloop::arclength::<f32, 2>(vtx2xy2.as_slice());
        assert!((arclen0 - arclen2).abs() < 1.0e-3);
        let edge2length2 = crate::polyloop::edge2length::<f32, 2>(vtx2xy2.as_slice());
        let min_edge_len2 = edge2length2
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!((min_edge_len2 - arclen2 / ndiv2 as f32).abs() < 1.0e-3);
    }
}

pub fn meshing_to_trimesh2<Index, Real>(
    vtxl2xy: &[Real],
    edge_length_boundary: Real,
    edge_length_internal: Real,
) -> (Vec<Index>, Vec<Real>)
where
    Real: nalgebra::RealField + Copy + 'static + num_traits::Float + AsPrimitive<usize>,
    Index: Copy + 'static,
    f64: AsPrimitive<Real>,
    usize: AsPrimitive<Real> + AsPrimitive<Index>,
{
    crate::trimesh2_dynamic::meshing_from_polyloop2::<Index, Real>(
        vtxl2xy,
        edge_length_boundary,
        edge_length_internal,
    )
}
