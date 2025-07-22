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
        *arrayref::array_ref!(affine, 0, 9),
        *arrayref::array_ref!(affine, 9, 3),
    );
    // dbg!(rot, transl);
    use del_geo_core::mat3_col_major::Mat3ColMajor;
    let tmp = rot.sub(&del_geo_core::mat3_col_major::from_identity()); // -2{n}{n}^T
    let (u, _s, _v) = del_geo_core::mat3_col_major::svd(
        &tmp,
        //del_geo_core::mat3_sym::EigenDecompositionModes::Analytic,
        del_geo_core::mat3_sym::EigenDecompositionModes::JacobiNumIter(100),
    )
    .unwrap();
    //dbg!(del_geo_core::mat3_col_major::determinant(&u));
    let (_u0, _u1, u2) = del_geo_core::mat3_col_major::to_columns(&u);
    let n = u2;
    use del_geo_core::vec3::Vec3;
    let p = n.scale(del_geo_core::vec3::dot(&transl, &u2) * 0.5);
    (n, p)
}

fn extract_triangles_in_symmetry(
    tri2flg: &mut [u8],
    i_tri_start: usize,
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    affine: &[f32; 12],
    tri2tri: &[usize],
) {
    assert!(i_tri_start < tri2vtx.len() / 3);
    let mut stack = vec![i_tri_start];
    while let Some(i_tri) = stack.pop() {
        if tri2flg[i_tri] != 0 {
            continue;
        }
        let cog = del_msh_cpu::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri).cog();
        let a_cog = del_geo_core::mat3x4_col_major::transform_affine(affine, &cog);
        //dbg!(cog, a_cog);
        // compute distance
        let dist_a = del_msh_cpu::trimesh3::distance_to_point3(tri2vtx, vtx2xyz, &a_cog);
        //dbg!(i_tri, dist_a);
        if dist_a > 0.03 {
            tri2flg[i_tri] = 1;
            continue;
        }
        tri2flg[i_tri] = 2;
        stack.push(tri2tri[i_tri * 3]);
        stack.push(tri2tri[i_tri * 3 + 1]);
        stack.push(tri2tri[i_tri * 3 + 2]);
    }
}

pub struct DetectedSymmetry {
    affine: [f32; 12],
    tris: Vec<usize>,
}

#[derive(Clone)]
pub struct Sample {
    xyz: [f32; 3],
    nrm: [f32; 3],
    i_tri: usize,
}

pub fn sym_detector(
    tri2vtx: &[usize],
    vtx2xyz: &[f32],
    i_seed: u64,
    num_sample: usize,
) -> Vec<DetectedSymmetry> {
    let tri2tri = del_msh_cpu::elem2elem::from_uniform_mesh(
        tri2vtx,
        3,
        &[0, 2, 4, 6],
        &[1, 2, 2, 0, 0, 1],
        vtx2xyz.len() / 3,
    );
    // use del_geo_core::vec3::Vec3;
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(i_seed);
    let tri2cumsumarea = del_msh_cpu::trimesh::tri2cumsumarea(tri2vtx, vtx2xyz, 3);
    let mut samples = vec![
        Sample {
            xyz: [0f32; 3],
            nrm: [0f32; 3],
            i_tri: 0
        };
        num_sample
    ];
    for i_sample in 0..num_sample {
        let r0 = rng.random::<f32>();
        let r1 = rng.random::<f32>();
        let (i_tri, r0, r1) = del_msh_cpu::trimesh::sample_uniformly(&tri2cumsumarea, r0, r1);
        let p0 = del_msh_cpu::trimesh::position_from_barycentric_coordinate::<_, 3>(
            tri2vtx, vtx2xyz, i_tri, r0, r1,
        );
        let n0 = del_msh_cpu::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri).unit_normal();
        samples[i_sample].xyz.copy_from_slice(&p0);
        samples[i_sample].nrm.copy_from_slice(&n0);
        samples[i_sample].i_tri = i_tri;
    }
    let mut pair2trans =
        Vec::<([f32; 12], usize, usize)>::with_capacity(num_sample * (num_sample - 1));
    for i_sample in 0..num_sample {
        let p_i = samples[i_sample].xyz;
        let n_i = samples[i_sample].nrm;
        for j_sample in i_sample + 1..num_sample {
            use del_geo_core::mat4_col_major;
            let p_j = samples[j_sample].xyz;
            let n_j = samples[j_sample].nrm;
            let pm = del_geo_core::edge3::position_from_ratio(&p_i, &p_j, 0.5);
            use del_geo_core::vec3::Vec3;
            let nrm = p_j.sub(&p_i).normalize();
            let r_mat = del_geo_core::mat3_col_major::sub(
                &del_geo_core::mat3_col_major::from_identity(),
                &del_geo_core::mat3_col_major::from_scaled_outer_product(2., &nrm, &nrm),
            );
            {
                let cos = del_geo_core::mat3_col_major::mult_vec(&r_mat, &n_i).dot(&n_j);
                if cos < 0.9 {
                    continue;
                } // filtering out
            }
            let t_mat = mat4_col_major::mult_three_mats_col_major(
                &mat4_col_major::from_translate(&pm),
                &mat4_col_major::from_mat3_col_major_adding_w(&r_mat, 1.0),
                &mat4_col_major::from_translate(&pm.scale(-1f32)),
            );
            {
                let p1_j = mat4_col_major::transform_homogeneous(&t_mat, &p_i).unwrap();
                assert!(p1_j.sub(&p_j).norm() < 1.0e-5);
                let p1_i = mat4_col_major::transform_homogeneous(&t_mat, &p_j).unwrap();
                assert!(p1_i.sub(&p_i).norm() < 1.0e-5);
            }
            let affine = del_geo_core::mat3x4_col_major::from_mat4_col_major(&t_mat);
            pair2trans.push((affine, i_sample, j_sample));
        }
    }
    println!("num pair: {}", pair2trans.len());
    let mut syms = Vec::<DetectedSymmetry>::new();
    for _iter in 0..30 {
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
            if sum_weight < sum_weight_pre * 1.01 {
                break;
            }
            sum_weight_pre = sum_weight_pre;
        }
        // dbg!(cur_trans, max_weight_and_pair);
        let (n, p) = get_normal_and_origin_from_affine_matrix_of_reflection(&cur_trans);
        dbg!(n, p);
        let i_tri = samples[max_weight_and_pair.1].i_tri;
        let j_tri = samples[max_weight_and_pair.2].i_tri;
        // region growing algorithm
        let mut tri2flg = vec![0; tri2vtx.len() / 3];
        extract_triangles_in_symmetry(&mut tri2flg, i_tri, tri2vtx, vtx2xyz, &cur_trans, &tri2tri);
        extract_triangles_in_symmetry(&mut tri2flg, j_tri, tri2vtx, vtx2xyz, &cur_trans, &tri2tri);
        // let num_tri = tri2flg.iter().filter(|&v| *v == 2 ).count();
        let tris: Vec<_> = tri2flg
            .iter()
            .enumerate()
            .filter(|(_i_tri, &v)| v == 2)
            .map(|(i_tri, _v)| i_tri)
            .collect();
        if tris.is_empty() {
            continue;
        }
        let ds = DetectedSymmetry {
            affine: cur_trans,
            tris,
        };
        syms.push(ds);
    }
    syms.sort_by(|a, b| a.tris.len().cmp(&b.tris.len()).reverse());
    syms
}

fn main() -> anyhow::Result<()> {
    use del_geo_core::vec3::Vec3;
    use rand::Rng;
    let (tri2vtx, vtx2xyz) = del_msh_cpu::io_obj::load_tri_mesh::<_, usize, f32>(
        "asset/spot/spot_triangulated.obj",
        None,
    )
    .unwrap();
    let vtx2xyz = {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        let rot = del_geo_core::mat3_col_major::from_bryant_angles(
            2.0 * std::f32::consts::PI * rng.random::<f32>(),
            2.0 * std::f32::consts::PI * rng.random::<f32>(),
            2.0 * std::f32::consts::PI * rng.random::<f32>(),
        );
        del_msh_cpu::vtx2xyz::transform_linear(&vtx2xyz, &rot)
    };
    let syms = sym_detector(&tri2vtx, &vtx2xyz, 9, 200);
    for (i_sym, sym) in syms.iter().enumerate() {
        let (n, p) = get_normal_and_origin_from_affine_matrix_of_reflection(&sym.affine);
        let (ex, ey) = del_geo_core::vec3::basis_xy_from_basis_z(&n);
        let (triq2vtxq, vtxq2xyz) = {
            // define square bi-sector plane
            use slice_of_array::SliceFlatExt;
            let vtxq2xyz = [
                p.sub(&ex).sub(&ey),
                p.add(&ex).sub(&ey),
                p.add(&ex).add(&ey),
                p.sub(&ex).add(&ey),
            ]
            .flat()
            .to_vec();
            let triq2vtxq = [[0usize, 1, 2], [0, 2, 3]].flat().to_vec();
            (triq2vtxq, vtxq2xyz)
        };
        let tris2vtx =
            del_msh_cpu::extract::from_uniform_mesh_from_list_of_elements(&tri2vtx, 3, &sym.tris);
        let mut trio2vtxo = vec![];
        let mut vtxo2xyz = vec![];
        del_msh_cpu::uniform_mesh::merge(&mut trio2vtxo, &mut vtxo2xyz, &triq2vtxq, &vtxq2xyz, 3);
        del_msh_cpu::uniform_mesh::merge(&mut trio2vtxo, &mut vtxo2xyz, &tris2vtx, &vtx2xyz, 3);
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
            format!("target/sym_{i_sym}.obj"),
            &trio2vtxo,
            &vtxo2xyz,
            3,
        )?;
    }
    Ok(())
}
