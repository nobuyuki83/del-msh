
fn kernel(a: &[f32; 12], b: &[f32; 12], h: f32) -> f32 {
    let dist = del_geo_core::vecn::squared_distance(a, b);
    let dist = dist / (h * h);
    if dist.abs() > 1. {
        0.
    } else {
        0.75 * (1. - dist)
    }
}

fn dw_kernel(a: &[f32; 12], x: &[f32; 12], h: f32) -> [f32; 12] {
    let dist = del_geo_core::vecn::distance(a, x);
    if dist.abs() > 1. {
        [0f32; 12]
    } else {
        use del_geo_core::vecn::VecN;
        a.scale(-1.5 * dist / (h * h))
    }
}

fn main() -> anyhow::Result<()> {
    use del_geo_core::vec3::Vec3;
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(6);
    let (tri2vtx, vtx2xyz) = del_msh_core::io_obj::load_tri_mesh::<_, usize, f32>(
        "asset/spot/spot_triangulated.obj",
        None,
    )?;
    let tri2cumsumarea = del_msh_core::trimesh::tri2cumsumarea(&tri2vtx, &vtx2xyz, 3);
    let num_vtxs = 300;
    let mut vtxs2xyz = vec![0f32; num_vtxs * 3];
    for i_vtxs in 0..num_vtxs {
        let r0 = rng.random::<f32>();
        let r1 = rng.random::<f32>();
        let (i_tri, r0, r1) =
            del_msh_core::sampling::sample_uniformly_trimesh(&tri2cumsumarea, r0, r1);
        let p0 = del_msh_core::trimesh::position_from_barycentric_coordinate::<_, 3>(
            &tri2vtx, &vtx2xyz, i_tri, r0, r1,
        );
        del_msh_core::vtx2xyz::to_vec3_mut(&mut vtxs2xyz, i_vtxs).copy_from_slice(&p0);
    }
    let mut pair2trans = Vec::<[f32; 12]>::with_capacity(num_vtxs * (num_vtxs - 1));
    for i_vtxs in 0..num_vtxs {
        let p_i = del_msh_core::vtx2xyz::to_vec3(&vtxs2xyz, i_vtxs);
        for j_vtxs in i_vtxs + 1..num_vtxs {
            let p_j = del_msh_core::vtx2xyz::to_vec3(&vtxs2xyz, j_vtxs);
            let pm = del_geo_core::edge3::position_from_ratio(p_i, p_j, 0.5);
            use del_geo_core::vec3::Vec3;
            let nrm = p_j.sub(p_i).normalize();
            let r_mat = del_geo_core::mat3_col_major::sub(
                &del_geo_core::mat3_col_major::from_identity(),
                &del_geo_core::mat3_col_major::from_scaled_outer_product(2., &nrm, &nrm),
            );
            use del_geo_core::mat4_col_major;
            let t_mat = mat4_col_major::mult_three_mats_col_major(
                &mat4_col_major::from_translate(&pm),
                &mat4_col_major::from_mat3_col_major_adding_w(&r_mat),
                &mat4_col_major::from_translate(&[-pm[0], -pm[1], -pm[2]]),
            );
            {
                let p1_j = mat4_col_major::transform_homogeneous(&t_mat, p_i).unwrap();
                assert!(p1_j.sub(p_j).norm() < 1.0e-5);
                let p1_i = mat4_col_major::transform_homogeneous(&t_mat, p_j).unwrap();
                assert!(p1_i.sub(p_i).norm() < 1.0e-5);
            }
            let affine = del_geo_core::mat3x4_col_major::from_mat4_col_major(&t_mat);
            pair2trans.push(affine);
        }
    }
    let mut cur_trans = {
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(1);
        let i_pair = rng.random_range(0..pair2trans.len());
        pair2trans[i_pair]
    };
    let window_size = 0.8;
    for itr in 0..100 {
        use del_geo_core::vecn::VecN;
        // compute density and its gradient
        let mut sum_weight = 0.;
        let mut sum_transf = [0f32; 12];
        for trans in &pair2trans {
            let w = kernel(&cur_trans, trans, window_size);
            // let dw = dw_kernel(&cur_trans, trans, window_size);
            sum_weight += w;
            sum_transf.add_in_place(&trans.scale(w));
        }
        cur_trans = sum_transf.scale(1.0/sum_weight);
        dbg!(cur_trans);
        dbg!(sum_weight);
    }
    Ok(())
}
