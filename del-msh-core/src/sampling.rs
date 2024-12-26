//! stochastic sampling on mesh

pub fn cumulative_areas_trimesh3_condition<F: Fn(usize) -> bool, Real>(
    tri2vtx: &[usize],
    vtx2xyz: &[Real],
    num_dim: usize,
    tri2isvalid: F,
) -> Vec<Real>
where
    Real: num_traits::Float + std::fmt::Debug + std::ops::MulAssign,
{
    assert!(num_dim == 2 || num_dim == 3);
    let num_tri = tri2vtx.len() / 3;
    assert_eq!(tri2vtx.len(), num_tri * 3);
    let mut cumulative_area_sum = Vec::<Real>::with_capacity(num_tri + 1);
    cumulative_area_sum.push(Real::zero());
    for idx_tri in 0..num_tri {
        let a0 = if !tri2isvalid(idx_tri) {
            Real::zero()
        } else if num_dim == 2 {
            crate::trimesh2::to_tri2(idx_tri, tri2vtx, vtx2xyz).area()
        } else {
            crate::trimesh3::to_tri3(tri2vtx, vtx2xyz, idx_tri).area()
        };
        let t0 = cumulative_area_sum[cumulative_area_sum.len() - 1];
        cumulative_area_sum.push(a0 + t0);
    }
    cumulative_area_sum
}

pub fn cumulative_area_sum<Real>(tri2vtx: &[usize], vtx2xyz: &[Real], num_dim: usize) -> Vec<Real>
where
    Real: num_traits::Float + std::fmt::Debug + std::ops::MulAssign,
{
    cumulative_areas_trimesh3_condition(tri2vtx, vtx2xyz, num_dim, |_itri| true)
}

/// sample points uniformly inside triangle mesh
/// * val01_a - uniformly sampled float value [0,1]
/// * val01_b - uniformly sampled float value [0,1]
/// # Return
/// (i_tri: usize, r0: Real, r1: Real)
pub fn sample_uniformly_trimesh<Real>(
    cumulative_area_sum: &[Real],
    val01_a: Real,
    val01_b: Real,
) -> (usize, Real, Real)
where
    Real: num_traits::Float + std::fmt::Debug,
{
    let (i_tri_l, r0, _p0) = crate::cumsum::sample(cumulative_area_sum, val01_a);
    if r0 + val01_b > Real::one() {
        let r0a = r0;
        let r1a = val01_b;
        return (i_tri_l, Real::one() - r1a, Real::one() - r0a);
    }
    (i_tri_l, r0, val01_b)
}

pub fn poisson_disk_sampling_from_polyloop2<RNG>(
    vtxl2xy: &[f32],
    radius: f32,
    num_iteration: usize,
    reng: &mut RNG,
) -> Vec<f32>
where
    RNG: rand::Rng,
{
    let (tri2vtx, vtx2xyz) =
        crate::trimesh2_dynamic::meshing_from_polyloop2::<usize, f32>(vtxl2xy, -1., -1.);
    let tri2cumarea = cumulative_area_sum(&tri2vtx, &vtx2xyz, 2);
    let mut vtx2vectwo: Vec<nalgebra::Vector2<f32>> = vec![];
    for _iter in 0..num_iteration {
        let (i_tri, r0, r1) = sample_uniformly_trimesh(&tri2cumarea, reng.gen(), reng.gen());
        let pos = crate::trimesh::position_from_barycentric_coordinate::<f32, 2>(
            &tri2vtx, &vtx2xyz, i_tri, r0, r1,
        );
        let pos = nalgebra::Vector2::<f32>::from_row_slice(&pos);
        let mut is_near = false;
        for pos0 in &vtx2vectwo {
            // TODO: use kd-tree to accelerate this process
            if (pos0 - pos).norm() > radius {
                continue;
            }
            is_near = true;
            break;
        }
        if is_near {
            continue;
        }
        vtx2vectwo.push(pos);
    }
    crate::vtx2xdim::from_array_of_nalgebra(&vtx2vectwo)
}

#[test]
fn test_poisson_disk_sampling() {
    let mut reng = rand::thread_rng();
    let vtxl2xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let vtx2xy = poisson_disk_sampling_from_polyloop2(&vtxl2xy, 0.1, 2000, &mut reng);
    {
        // write boundary and
        let mut vtxl2xy = vtxl2xy.clone();
        vtxl2xy.extend(vtx2xy);
        let _ = crate::io_obj::save_edge2vtx_vtx2xyz(
            "target/poisson_disk.obj",
            &[0, 1, 1, 2, 2, 3, 3, 0],
            &vtxl2xy,
            2,
        );
    }
}
