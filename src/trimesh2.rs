//! methods for 2D triangle mesh


// --------------------------
// below: vtx2***

use num_traits::AsPrimitive;

#[allow(clippy::identity_op)]
pub fn vtx2area<Real>(
    tri2vtx: &[usize],
    vtx2xy: &[Real]) -> Vec<Real>
    where Real: num_traits::Float + std::ops::AddAssign
{
    let num_vtx = vtx2xy.len() / 2;
    assert_eq!(vtx2xy.len(), num_vtx*2);
    let mut vtx2area = vec!(Real::zero(); num_vtx);
    let one_third = Real::one() / (Real::one() + Real::one() + Real::one());
    for node2vtx in tri2vtx.chunks(3) {
        let (i0,i1,i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = &vtx2xy[i0 * 2..i0 * 2 + 2].try_into().unwrap();
        let p1 = &vtx2xy[i1 * 2..i1 * 2 + 2].try_into().unwrap();
        let p2 = &vtx2xy[i2 * 2..i2 * 2 + 2].try_into().unwrap();
        let a0 = del_geo::tri2::area_(p0,p1,p2) * one_third;
        vtx2area[i0] += a0;
        vtx2area[i1] += a0;
        vtx2area[i2] += a0;
    }
    vtx2area
}

// ---------------
// below: tri2***

#[allow(clippy::identity_op)]
pub fn tri2area(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> Vec<f32>
{
    let mut tri2area = Vec::<f32>::with_capacity(tri2vtx.len() / 3);
    for node2vtx in tri2vtx.chunks(3) {
        let (i0,i1,i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = vtx2xyz[i0 * 2 + 0..i0 * 2 + 2].try_into().unwrap();
        let p1 = vtx2xyz[i1 * 2 + 0..i1 * 2 + 2].try_into().unwrap();
        let p2 = vtx2xyz[i2 * 2 + 0..i2 * 2 + 2].try_into().unwrap();
        let area = del_geo::tri2::area_(p0, p1, p2);
        tri2area.push(area);
    }
    tri2area
}

#[allow(clippy::identity_op)]
pub fn tri2circumcenter(
    tri2vtx: &[usize],
    vtx2xyz: &[f32]) -> Vec<f32>
{
    let mut tri2cc = Vec::<f32>::with_capacity(tri2vtx.len() );
    for node2vtx in tri2vtx.chunks(3) {
        let (i0, i1, i2) = (node2vtx[0], node2vtx[1], node2vtx[2]);
        let p0 = nalgebra::Vector2::<f32>::from_row_slice(&vtx2xyz[i0 * 2 + 0..i0 * 2 + 2]);
        let p1 = nalgebra::Vector2::<f32>::from_row_slice(&vtx2xyz[i1 * 2 + 0..i1 * 2 + 2]);
        let p2 = nalgebra::Vector2::<f32>::from_row_slice(&vtx2xyz[i2 * 2 + 0..i2 * 2 + 2]);
        let cc = del_geo::tri2::circumcenter(&p0, &p1, &p2);
        tri2cc.push(cc[0]);
        tri2cc.push(cc[1]);
    }
    tri2cc
}

pub fn search_bruteforce_one_triangle_include_input_point<Index, Real>(
    q: &[Real;2],
    tri2vtx: &[Index],
    vtx2xy: &[Real]) -> Option<(usize, Real, Real)>
    where Real: num_traits::Float,
    Index: 'static + Copy + AsPrimitive<usize>
{
    for (i_tri, node2vtx) in tri2vtx.chunks(3).enumerate() {
        let (i0, i1, i2)
            = (node2vtx[0].as_(), node2vtx[1].as_(), node2vtx[2].as_());
        let p0 = vtx2xy[i0*2..i0*2+2].try_into().unwrap();
        let p1 = vtx2xy[i1*2..i1*2+2].try_into().unwrap();
        let p2 = vtx2xy[i2*2..i2*2+2].try_into().unwrap();
        let a0 = del_geo::tri2::area_(q, p1, p2);
        if a0 < Real::zero() { continue; }
        let a1 = del_geo::tri2::area_(q, p2, p0);
        if a1 < Real::zero() { continue; }
        let a2 = del_geo::tri2::area_(q, p0, p1);
        if a2 < Real::zero() { continue; }
        let sum_area_inv = Real::one() / (a0 + a1 + a2);
        return Some((i_tri, a0*sum_area_inv, a1*sum_area_inv));
    }
    None
}

// ------------------------------
// below: nalgebra dependent

#[allow(clippy::identity_op)]
pub fn area_of_a_triangle<Real>(
    tri2vtx: &[usize],
    vtx2vectwo: &[nalgebra::Vector2<Real>],
    i_tri: usize) -> Real
    where Real: nalgebra::RealField + Copy
{
    let i0 = tri2vtx[i_tri*3+0];
    let i1 = tri2vtx[i_tri*3+1];
    let i2 = tri2vtx[i_tri*3+2];
    del_geo::tri2::area(&vtx2vectwo[i0], &vtx2vectwo[i1], &vtx2vectwo[i2])
}

// -----------------------------


#[test]
fn test_volonoi() {
    let vtxl2xy = vec!(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let (tri2vtx, vtx2xy)
        = crate::trimesh2_dynamic::meshing_from_polyloop2::<usize, f32>(
        &vtxl2xy, 0.08, 0.08);
    let tri2xycc = crate::trimesh2::tri2circumcenter(&tri2vtx, &vtx2xy);
    let (bedge2vtx, tri2triedge)
        = crate::trimesh_topology::boundaryedge2vtx(&tri2vtx, vtx2xy.len()/2);
    //
    let bedge2xymp = {
        let mut bedge2xymp = vec!(0f32; bedge2vtx.len());
        for (i_bedge, node2vtx) in bedge2vtx.chunks(2).enumerate() {
            let (i0_vtx, i1_vtx) = (node2vtx[0], node2vtx[1]);
            bedge2xymp[i_bedge*2+0] = (vtx2xy[i0_vtx*2+0] + vtx2xy[i1_vtx*2+0])*0.5;
            bedge2xymp[i_bedge*2+1] = (vtx2xy[i0_vtx*2+1] + vtx2xy[i1_vtx*2+1])*0.5;
        }
        bedge2xymp
    };
    let pnt2xy = {
        let mut pnt2xy = tri2xycc;
        pnt2xy.extend(bedge2xymp);
        pnt2xy
    };
    let vedge2pnt = {
        let mut vedge2pnt = vec!(0usize;0);
        for (i_tri, node2triedge) in tri2triedge.chunks(3).enumerate() {
            for i_node in 0..3 {
                let i_triedge = node2triedge[i_node];
                if i_triedge <= i_tri { continue; }
                vedge2pnt.extend(&[i_tri, i_triedge]);
            }
        }
        let (face2idx, idx2node)
            = crate::elem2elem::face2node_of_simplex_element(2);
        let bedge2bedge = crate::elem2elem::from_uniform_mesh(
            &bedge2vtx, 2,
            &face2idx, &idx2node, vtx2xy.len()/2);
        let num_tri = tri2vtx.len() / 3;
        for (i_bedge, node2bedge) in bedge2bedge.chunks(2).enumerate() {
            for i_node in 0..2 {
                let j_bedge = node2bedge[i_node];
                if i_bedge > j_bedge { continue; }
                vedge2pnt.extend(&[i_bedge + num_tri, j_bedge + num_tri]);
            }
        }
        vedge2pnt
    };
    let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
        "target/volonoi.obj", &vedge2pnt, &pnt2xy, 2);
}

