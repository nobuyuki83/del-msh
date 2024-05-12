//! stochastic sampling on mesh

use rand::Rng;

#[allow(clippy::identity_op)]
pub fn cumulative_areas_trimesh3_condition<F: Fn(usize) -> bool, Real>(
    tri2vtx: &[usize],
    vtx2xyz: &[Real],
    num_dim: usize,
    tri2isvalid: F) -> Vec<Real>
where Real: num_traits::Float
{
    assert!(num_dim == 2 || num_dim == 3);
    let num_tri = tri2vtx.len() / 3;
    assert_eq!(tri2vtx.len(), num_tri * 3);
    let mut cumulative_area_sum = Vec::<Real>::with_capacity(num_tri + 1);
    cumulative_area_sum.push(Real::zero());
    for idx_tri in 0..num_tri {
        let a0 = if !tri2isvalid(idx_tri) {
            Real::zero()
        } else {
            let i0 = tri2vtx[idx_tri * 3 + 0];
            let i1 = tri2vtx[idx_tri * 3 + 1];
            let i2 = tri2vtx[idx_tri * 3 + 2];
            if num_dim == 2 {
                del_geo::tri2::area_(
                    &vtx2xyz[i0 * 2 + 0..i0 * 2 + 2].try_into().unwrap(),
                    &vtx2xyz[i1 * 2 + 0..i1 * 2 + 2].try_into().unwrap(),
                    &vtx2xyz[i2 * 2 + 0..i2 * 2 + 2].try_into().unwrap())
            }
            else{
                del_geo::tri3::area_(
                    &vtx2xyz[i0 * 3 + 0..i0 * 3 + 3].try_into().unwrap(),
                    &vtx2xyz[i1 * 3 + 0..i1 * 3 + 3].try_into().unwrap(),
                    &vtx2xyz[i2 * 3 + 0..i2 * 3 + 3].try_into().unwrap())
            }
        };
        let t0 = cumulative_area_sum[cumulative_area_sum.len() - 1];
        cumulative_area_sum.push(a0 + t0);
    }
    cumulative_area_sum
}


pub fn cumulative_area_sum<Real>(
    tri2vtx: &[usize],
    vtx2xyz: &[Real],
    num_dim: usize) -> Vec<Real>
    where Real: num_traits::Float
{
    cumulative_areas_trimesh3_condition(
        tri2vtx, vtx2xyz,  num_dim, |_itri| { true })
}

/// sample points uniformly inside triangle mesh
/// * val01_a - uniformly sampled float value [0,1]
pub fn sample_uniformly_trimesh<Real>(
    cumulative_area_sum: &[Real],
    val01_a: Real,
    val01_b: Real) -> (usize, Real, Real)
where Real: num_traits::Float
{
    let num_tri = cumulative_area_sum.len() - 1;
    let a0 = val01_a * cumulative_area_sum[num_tri];
    let mut i_tri_l = 0;
    let mut i_tri_u = num_tri;
    loop {  // bisection method
        assert!(cumulative_area_sum[i_tri_l] < a0);
        assert!(a0 <= cumulative_area_sum[i_tri_u]);
        let i_tri_h = (i_tri_u + i_tri_l) / 2;
        if i_tri_u - i_tri_l == 1 { break; }
        if cumulative_area_sum[i_tri_h] < a0 {
            i_tri_l = i_tri_h;
        } else {
            i_tri_u = i_tri_h;
        }
    }
    assert!(cumulative_area_sum[i_tri_l] < a0);
    assert!(a0 <= cumulative_area_sum[i_tri_l + 1]);
    let r0 = (a0 - cumulative_area_sum[i_tri_l]) / (cumulative_area_sum[i_tri_l + 1] - cumulative_area_sum[i_tri_l]);
    if r0 + val01_b > Real::one() {
        let r0a = r0;
        let r1a = val01_b;
        return (i_tri_l, Real::one() - r1a, Real::one() - r0a);
    }
    (i_tri_l, r0, val01_b)
}

pub fn poisson_disk_sampling_from_polyloop2(
    vtxl2xy: &[f32],
    radius: f32,
    num_iteration: usize,
) -> Vec<f32>
{
    let (tri2vtx, vtx2xyz)
        = crate::trimesh2_dynamic::meshing_from_polyloop2::<usize, f32>(
        vtxl2xy, -1., -1.);
    let tri2cumarea = cumulative_area_sum(&tri2vtx, &vtx2xyz, 2);
    let mut rng = rand::thread_rng();
    let mut vtx2vectwo = vec!(nalgebra::Vector2::<f32>::zeros();0);
    for _iter in 0..num_iteration {
        let (i_tri, r0, r1)
            = sample_uniformly_trimesh(&tri2cumarea, rng.gen(), rng.gen());
        let pos = crate::trimesh::position_from_barycentric_coordinate::<f32, 2>(
            &tri2vtx, &vtx2xyz, i_tri, r0, r1);
        let pos = nalgebra::Vector2::<f32>::from_row_slice(&pos);
        let mut is_near = false;
        for pos0 in &vtx2vectwo { // TODO: use kd-tree to accelerate this process
            if (pos0 - pos).norm() > radius { continue; }
            is_near = true;
            break;
        }
        if is_near { continue; }
        vtx2vectwo.push(pos);
    }
    crate::vtx2xyz::from_array_of_nalgebra(&vtx2vectwo)
}

#[test]
fn test_poisson_disk_sampling() {
    let vtxl2xy = vec!(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    let vtx2xy = poisson_disk_sampling_from_polyloop2(
        &vtxl2xy, 0.1, 2000);
    {  // write boundary and
        let mut vtxl2xy = vtxl2xy.clone();
        vtxl2xy.extend(vtx2xy);
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/poisson_disk.obj",
            &[0, 1, 1, 2, 2, 3, 3, 0], &vtxl2xy, 2);
    }
}

