

fn kernel(a: &[f32; 12], b: &[f32; 12], h: f32) -> f32 {
    let dist = del_geo_core::vecn::squared_distance(a, b);
    let dist = dist / (h * h);
    if dist.abs() > 1. {
        0.
    } else {
        0.75 * (1. - dist)
    }
}

fn get_normal_and_origin_from_affine_matrix_of_reflection(
    affine: &[f32; 12],
) -> ([f32; 3], [f32; 3]) {
    let (rot, transl): ([f32; 9], [f32; 3]) = (
        arrayref::array_ref!(affine, 0, 9).clone(),
        arrayref::array_ref!(affine, 9, 3).clone(),
    );
    // dbg!(rot, transl);
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    let tmp = rot.sub(&del_geo_core::mat3_col_major::from_identity()); // -2{n}{n}^T
    let (u, s, v) = del_geo_core::mat3_col_major::svd(
        &tmp,
        //del_geo_core::mat3_sym::EigenDecompositionModes::Analytic,
        del_geo_core::mat3_sym::EigenDecompositionModes::JacobiNumIter(100),
    )
    .unwrap();
    //dbg!(del_geo_core::mat3_col_major::determinant(&u));
    let (u0, u1, u2) = del_geo_core::mat3_col_major::to_columns(&u);
    let n = u2;
    use del_geo_core::vec3::Vec3;
    let p = n.scale(del_geo_core::vec3::dot(&transl, &u2) * 0.5);
    (n, p)
}

pub fn sym_detector(
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    i_seed: u64,
    num_sample: usize,
) -> [f32; 12] {
    // use del_geo_core::vec3::Vec3;
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_seed);
    let tri2cumsumarea = del_msh_core::trimesh::tri2cumsumarea(&tri2vtx, &vtx2xyz, 3);
    let mut sample2xyz = vec![0f32; num_sample * 3];
    let mut sample2nrm = vec![0f32; num_sample * 3];
    for i_sample in 0..num_sample {
        let r0 = rng.random::<f32>();
        let r1 = rng.random::<f32>();
        let (i_tri, r0, r1) =
            del_msh_core::sampling::sample_uniformly_trimesh(&tri2cumsumarea, r0, r1);
        let p0 = del_msh_core::trimesh::position_from_barycentric_coordinate::<_, 3>(
            &tri2vtx, &vtx2xyz, i_tri, r0, r1,
        );
        let n0 = del_msh_core::trimesh3::to_tri3(&tri2vtx, &vtx2xyz, i_tri).unit_normal();
        del_msh_core::vtx2xyz::to_vec3_mut(&mut sample2xyz, i_sample).copy_from_slice(&p0);
        del_msh_core::vtx2xyz::to_vec3_mut(&mut sample2nrm, i_sample).copy_from_slice(&n0);
    }
    let mut pair2trans = Vec::<([f32; 12], usize, usize)>::with_capacity(num_sample * (num_sample - 1));
    for i_sample in 0..num_sample {
        let p_i = del_msh_core::vtx2xyz::to_vec3(&sample2xyz, i_sample);
        let n_i = del_msh_core::vtx2xyz::to_vec3(&sample2nrm, i_sample);
        for j_sample in i_sample + 1..num_sample {
            use del_geo_core::mat4_col_major;
            let p_j = del_msh_core::vtx2xyz::to_vec3(&sample2xyz, j_sample);
            let n_j = del_msh_core::vtx2xyz::to_vec3(&sample2nrm, j_sample);
            let pm = del_geo_core::edge3::position_from_ratio(p_i, p_j, 0.5);
            use del_geo_core::vec3::Vec3;
            let nrm = p_j.sub(p_i).normalize();
            let r_mat = del_geo_core::mat3_col_major::sub(
                &del_geo_core::mat3_col_major::from_identity(),
                &del_geo_core::mat3_col_major::from_scaled_outer_product(2., &nrm, &nrm),
            );
            {
                let cos = del_geo_core::mat3_col_major::mult_vec(&r_mat, &n_i).dot(&n_j);
                if cos < 0.9 { continue; } // filtering out
            }
            let t_mat = mat4_col_major::mult_three_mats_col_major(
                &mat4_col_major::from_translate(&pm),
                &mat4_col_major::from_mat3_col_major_adding_w(&r_mat),
                &mat4_col_major::from_translate(&pm.scale(-1f32)),
            );
            {
                let p1_j = mat4_col_major::transform_homogeneous(&t_mat, p_i).unwrap();
                assert!(p1_j.sub(p_j).norm() < 1.0e-5);
                let p1_i = mat4_col_major::transform_homogeneous(&t_mat, p_j).unwrap();
                assert!(p1_i.sub(p_i).norm() < 1.0e-5);
            }
            let affine = del_geo_core::mat3x4_col_major::from_mat4_col_major(&t_mat);
            pair2trans.push((affine, i_sample, j_sample));
        }
    }
    println!("num pair: {}", pair2trans.len());
    for _iter in 0..10 {
        let (mut cur_trans, _, _) = {
            let i_pair = rng.random_range(0..pair2trans.len());
            pair2trans[i_pair]
        };
        let window_size = 1.0;
        let mut sum_weight_pre = 0f32;
        let mut max_weight_and_pair = (0f32, usize::MAX, usize::MAX);
        for _itr in 0..100 {
            // TODO: accelerate using Kd-tree
            use del_geo_core::vecn::VecN;
            let mut sum_weight = 0.;
            let mut sum_transf = [0f32; 12];
            for (trans, i_sample, j_sample) in &pair2trans {
                let w = kernel(&cur_trans, trans, window_size);
                sum_weight += w;
                sum_transf.add_in_place(&trans.scale(w));
                if w > max_weight_and_pair.0 {
                    max_weight_and_pair = (w, *i_sample, *j_sample);
                }
            }
            cur_trans = sum_transf.scale(1.0 / sum_weight);
            if sum_weight < sum_weight_pre * 1.01 { break;}
            sum_weight_pre = sum_weight_pre;
        }
        dbg!(cur_trans, max_weight_and_pair);
        // TODO: region grow algorithm
    }
    [0f32;12]
}

fn main() -> anyhow::Result<()> {
    use del_geo_core::vec3::Vec3;
    let (tri2vtx, vtx2xyz) = del_msh_core::io_obj::load_tri_mesh::<_, usize, f32>(
        "asset/spot/spot_triangulated.obj",
        None,
    )
        .unwrap();
    let affine = sym_detector(&tri2vtx, &vtx2xyz, 9, 200);
    dbg!(affine);
    let (n, p) = get_normal_and_origin_from_affine_matrix_of_reflection(&affine);
    dbg!(n, p);
    let (ex, ey) = del_geo_core::vec3::basis_xy_from_basis_z(&n);
    use slice_of_array::SliceFlatExt;
    let vtx2xyz = [
        p.sub(&ex).sub(&ey),
        p.add(&ex).sub(&ey),
        p.add(&ex).add(&ey),
        p.sub(&ex).add(&ey),
    ]
    .flat()
    .to_vec();
    let tri2vtx = [[0usize, 1, 2], [0, 2, 3]].flat().to_vec();
    del_msh_core::io_obj::save_tri2vtx_vtx2xyz("hoge.obj", &tri2vtx, &vtx2xyz, 3)?;
    Ok(())
}
