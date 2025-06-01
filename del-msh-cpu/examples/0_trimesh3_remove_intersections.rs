fn make_toy_problem(fpath_start: &str, fpath_goal: &str) {
    let (tri2vtx, vtx2xyz) = {
        let (tri2vtx, mut vtx2xyz) =
            del_msh_cpu::trimesh3_primitive::capsule_yup(0.3, 1.0, 16, 8, 16);
        // adding little noise (my program has a bug when there is co-planer triangles)
        use rand::Rng;
        use rand::SeedableRng;
        let mut reng = rand_chacha::ChaChaRng::seed_from_u64(0u64);
        vtx2xyz
            .iter_mut()
            .for_each(|v| *v += (reng.random::<f64>() - 0.5) * 0.01);
        (tri2vtx, vtx2xyz)
    };
    {
        let rot_x_90 = del_geo_core::mat4_col_major::from_bryant_angles::<f64>(1.5, 0.1, 0.1);
        let transl_x = del_geo_core::mat4_col_major::from_translate::<f64>(&[1.0, 0.1, 0.1]);
        let transf = del_geo_core::mat4_col_major::mult_mat_col_major(&rot_x_90, &transl_x);
        let mut vtx2xyz0 = vtx2xyz.as_slice().to_vec();
        let vtx2xyz1 = del_msh_cpu::vtx2xyz::transform_homogeneous(
            vtx2xyz.as_slice(),
            arrayref::array_ref![transf.as_slice(), 0, 16],
        );
        let mut tri2vtx0 = tri2vtx.as_slice().to_vec();
        let tri2vtx1 = tri2vtx.as_slice().to_vec();
        del_msh_cpu::uniform_mesh::merge(&mut tri2vtx0, &mut vtx2xyz0, &tri2vtx1, &vtx2xyz1, 3);
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(fpath_start, &tri2vtx0, &vtx2xyz0, 3).unwrap()
    }
    {
        let rot_x_90 = del_geo_core::mat4_col_major::from_bryant_angles::<f64>(1.5, 0.0, 0.0);
        let transl_x = del_geo_core::mat4_col_major::from_translate::<f64>(&[0.5, 0.0, 0.0]);
        let transf = del_geo_core::mat4_col_major::mult_mat_col_major(&rot_x_90, &transl_x);
        let mut vtx2xyz0 = vtx2xyz.as_slice().to_vec();
        let vtx2xyz1 = del_msh_cpu::vtx2xyz::transform_homogeneous(
            vtx2xyz.as_slice(),
            arrayref::array_ref![transf.as_slice(), 0, 16],
        );
        let mut tri2vtx0 = tri2vtx.as_slice().to_vec();
        let tri2vtx1 = tri2vtx.as_slice().to_vec();
        del_msh_cpu::uniform_mesh::merge(&mut tri2vtx0, &mut vtx2xyz0, &tri2vtx1, &vtx2xyz1, 3);
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(fpath_goal, &tri2vtx0, &vtx2xyz0, 3).unwrap()
    }
}

fn main() -> anyhow::Result<()> {
    let obj_file_path_start = "target/del_msh_0__start.obj";
    let obj_file_path_goal = "target/del_msh_0__goal.obj";
    make_toy_problem(obj_file_path_start, obj_file_path_goal); // generate two .obj files in "del-mesh/target" directory
                                                               //
    let (tri2vtx, vtx2xyz_start) =
        del_msh_cpu::io_obj::load_tri_mesh::<_, usize, f64>(obj_file_path_start, None)?;
    /*
    {
        dbg!(tri2vtx.len()/3);
        let mut tri2tri_new = vec!(usize::MAX; tri2vtx.len()/3);
        tri2tri_new[1307] = 0;
        tri2tri_new[1339] = 1;
        let (tri2vtx_new, num_vtx_new, vtx2vtx_new)
            = del_msh_cpu::extract::extract(&tri2vtx, vtx2xyz_start.len()/3, &tri2tri_new, 2);
        let vtx2xyz_new = map_values_old2new(&vtx2xyz_start, &vtx2vtx_new, num_vtx_new, 3);
        del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz("target/debug.obj", &tri2vtx_new, &vtx2xyz_new, 3)?;
    }
     */

    let (_tri2vtx, vtx2xyz_goal) =
        del_msh_cpu::io_obj::load_tri_mesh::<_, usize, f64>(obj_file_path_goal, None)?;
    assert_eq!(tri2vtx, _tri2vtx);
    let vtx2xyz_out =
        del_msh_cpu::trimesh3_move_avoid_intersection::match_vtx2xyz_while_avoid_collision(
            &tri2vtx,
            &vtx2xyz_start,
            &vtx2xyz_goal,
            del_msh_cpu::trimesh3_move_avoid_intersection::Params {
                k_diff: 100.0,
                k_contact: 1000.0,
                dist0: 0.01,
                alpha: 0.01,
                num_iter: 20,
            },
        );
    del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
        "target/del_msh_0__out.obj",
        &tri2vtx,
        &vtx2xyz_out,
        3,
    )?;
    Ok(())
}
