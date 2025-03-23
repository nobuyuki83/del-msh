//! methods for 2D triangle mesh

// --------------------------
// below: vtx2***

use num_traits::AsPrimitive;

pub fn vtx2area<Real>(tri2vtx: &[usize], vtx2xy: &[Real]) -> Vec<Real>
where
    Real: num_traits::Float + std::ops::AddAssign,
{
    let num_vtx = vtx2xy.len() / 2;
    assert_eq!(vtx2xy.len(), num_vtx * 2);
    let mut vtx2area = vec![Real::zero(); num_vtx];
    let one_third = Real::one() / (Real::one() + Real::one() + Real::one());
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = arrayref::array_ref![vtx2xy, i0 * 2, 2];
        let p1 = arrayref::array_ref![vtx2xy, i1 * 2, 2];
        let p2 = arrayref::array_ref![vtx2xy, i2 * 2, 2];
        let a0 = del_geo_core::tri2::area(p0, p1, p2) * one_third;
        vtx2area[i0] += a0;
        vtx2area[i1] += a0;
        vtx2area[i2] += a0;
    }
    vtx2area
}

// ---------------
// below: tri2***

pub fn tri2area(tri2vtx: &[usize], vtx2xyz: &[f32]) -> Vec<f32> {
    let num_tri = tri2vtx.len() / 3;
    let mut tri2area = Vec::<f32>::with_capacity(num_tri);
    for i_tri in 0..num_tri {
        let area = to_tri2(i_tri, tri2vtx, vtx2xyz).area();
        tri2area.push(area);
    }
    tri2area
}

pub fn tri2circumcenter(tri2vtx: &[usize], vtx2xyz: &[f32]) -> Vec<f32> {
    let mut tri2cc = Vec::<f32>::with_capacity(tri2vtx.len());
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0: &[f32; 2] = crate::vtx2xy::to_vec2(vtx2xyz, i0);
        let p1: &[f32; 2] = crate::vtx2xy::to_vec2(vtx2xyz, i1);
        let p2: &[f32; 2] = crate::vtx2xy::to_vec2(vtx2xyz, i2);
        let cc = del_geo_core::tri2::circumcenter(p0, p1, p2);
        tri2cc.push(cc[0]);
        tri2cc.push(cc[1]);
    }
    tri2cc
}

pub fn search_bruteforce_one_triangle_include_input_point<Index, Real>(
    q: &[Real; 2],
    tri2vtx: &[Index],
    vtx2xy: &[Real],
) -> Option<(usize, Real, Real)>
where
    Real: num_traits::Float,
    Index: 'static + Copy + AsPrimitive<usize>,
{
    for (i_tri, node2vtx) in tri2vtx.chunks(3).enumerate() {
        let (i0, i1, i2) = (node2vtx[0].as_(), node2vtx[1].as_(), node2vtx[2].as_());
        let p0 = arrayref::array_ref!(vtx2xy, i0 * 2, 2);
        let p1 = arrayref::array_ref!(vtx2xy, i1 * 2, 2);
        let p2 = arrayref::array_ref!(vtx2xy, i2 * 2, 2);
        let Some((r0, r1)) = del_geo_core::tri2::is_inside(p0, p1, p2, q, Real::one()) else {
            continue;
        };
        return Some((i_tri, r0, r1));
    }
    None
}

pub fn to_corner_points<Index, Real>(
    tri2vtx: &[Index],
    vtx2xy: &[Real],
    i_tri: usize,
) -> ([Real; 2], [Real; 2], [Real; 2])
where
    Real: Copy,
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    let i_tri: usize = i_tri.as_();
    let i0: usize = tri2vtx[i_tri * 3].as_();
    let i1: usize = tri2vtx[i_tri * 3 + 1].as_();
    let i2: usize = tri2vtx[i_tri * 3 + 2].as_();
    (
        [vtx2xy[i0 * 2], vtx2xy[i0 * 2 + 1]],
        [vtx2xy[i1 * 2], vtx2xy[i1 * 2 + 1]],
        [vtx2xy[i2 * 2], vtx2xy[i2 * 2 + 1]],
    )
}

/// the center is the origin
pub fn from_circle(rad: f32, n: usize) -> (Vec<usize>, Vec<f32>) {
    let num_vtx = n + 1;
    let mut vtx2xy = Vec::<f32>::with_capacity(num_vtx * 2);
    vtx2xy.push(0f32);
    vtx2xy.push(0f32);
    for i in 0..n {
        let theta = std::f32::consts::PI * 2_f32 * i as f32 / n as f32;
        vtx2xy.push(rad * theta.cos());
        vtx2xy.push(rad * theta.sin());
    }
    let mut tri2vtx = Vec::<usize>::with_capacity(n * 3);
    for i in 0..n {
        tri2vtx.push(0);
        tri2vtx.push(i + 1);
        tri2vtx.push((i + 1) % n + 1);
    }
    (tri2vtx, vtx2xy)
}

// ------------------------------
// below: nalgebra dependent

pub fn area_of_a_triangle<Real>(tri2vtx: &[usize], vtx2vectwo: &[[Real; 2]], i_tri: usize) -> Real
where
    Real: num_traits::Float + Copy,
{
    let i0 = tri2vtx[i_tri * 3];
    let i1 = tri2vtx[i_tri * 3 + 1];
    let i2 = tri2vtx[i_tri * 3 + 2];
    del_geo_core::tri2::area(&vtx2vectwo[i0], &vtx2vectwo[i1], &vtx2vectwo[i2])
}

// -----------------------------

pub fn to_tri2<'a, Index, Real>(
    i_tri: usize,
    tri2vtx: &'a [Index],
    vtx2xy: &'a [Real],
) -> del_geo_core::tri2::Tri2<'a, Real>
where
    Index: AsPrimitive<usize>,
{
    let i0: usize = tri2vtx[i_tri * 3].as_();
    let i1: usize = tri2vtx[i_tri * 3 + 1].as_();
    let i2: usize = tri2vtx[i_tri * 3 + 2].as_();
    del_geo_core::tri2::Tri2 {
        p0: arrayref::array_ref!(vtx2xy, i0 * 2, 2),
        p1: arrayref::array_ref!(vtx2xy, i1 * 2, 2),
        p2: arrayref::array_ref!(vtx2xy, i2 * 2, 2),
    }
}
