


fn hoge(path: String, vtx2xyz: &[f32], vt2lhs0: &[f32]) {
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vt2lhs0.len(), num_vtx*3);
    // draw edges
    let edge2vtxe = (0..num_vtx)
        .flat_map(|i| [i * 2, i * 2 + 1])
        .collect::<Vec<usize>>();
    let mut vtxe2xyz = vec![0f32; vtx2xyz.len() * 2];
    for i_vtx in 0..num_vtx {
        vtxe2xyz[i_vtx * 6 + 0] = vtx2xyz[i_vtx * 3 + 0];
        vtxe2xyz[i_vtx * 6 + 1] = vtx2xyz[i_vtx * 3 + 1];
        vtxe2xyz[i_vtx * 6 + 2] = vtx2xyz[i_vtx * 3 + 2];
        vtxe2xyz[i_vtx * 6 + 3] = vtx2xyz[i_vtx * 3 + 0] + vt2lhs0[i_vtx * 3 + 0];
        vtxe2xyz[i_vtx * 6 + 4] = vtx2xyz[i_vtx * 3 + 1] + vt2lhs0[i_vtx * 3 + 1];
        vtxe2xyz[i_vtx * 6 + 5] = vtx2xyz[i_vtx * 3 + 2] + vt2lhs0[i_vtx * 3 + 2];
    }
    del_msh_cpu::io_wavefront_obj::save_edge2vtx_vtx2xyz(
        path,
        &edge2vtxe,
        &vtxe2xyz,
        3,
    )
        .unwrap();
}

fn main() {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::io_wavefront_obj::load_tri_mesh::<_, u32, f32>(
        "asset/spot/spot_triangulated.obj",
        None,
    )
    .unwrap();
    let num_vtx = vtx2xyz.len() / 3;
    let vtx2rhs = {
        let mut grad0 = vec![0f32; vtx2xyz.len()];
        /*
        grad0[0] = 1.0;
        grad0[1] = 0.0;
        grad0[2] = 0.0;
         */
        use rand::{SeedableRng, Rng};
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        grad0.iter_mut().for_each(|v| *v = rng.random::<f32>() * 2.0 - 1.0 );
        grad0
    };
    let spoisson = del_msh_cpu::nbody::ScreenedPoison::new(10.0, 1.0e-3);
    let vtx2lhs0 = {
        let mut vtx2lhs = vec![0f32; vtx2xyz.len()];
        del_msh_cpu::nbody::screened_poisson3(
            &spoisson,
            &vtx2xyz,
            &mut vtx2lhs,
            &vtx2xyz,
            &vtx2rhs,
        );
        vtx2lhs
    };
    let vtx2lhs1 = {
        let max_depth = 10;
        let transform_world2unit= {
            let aabb3 = del_msh_cpu::vtx2xyz::aabb3(&vtx2xyz, 0.);
            let scale_unit2world = del_geo_core::aabb3::max_edge_size(&aabb3);
            let scale_world2unit = 1.0 / scale_unit2world;
            let center = del_geo_core::aabb3::center(&aabb3);
            use del_geo_core::{mat4_col_major::Mat4ColMajor, vec3::Vec3};
            let m1 = del_geo_core::mat4_col_major::from_translate(&center.scale(-1.));
            let m2 = del_geo_core::mat4_col_major::from_scale_uniform(scale_world2unit);
            let m3 = del_geo_core::mat4_col_major::from_translate(&[0.5, 0.5, 0.5]);
            m3.mult_mat(&m2).mult_mat(&m1)
        };
        let mut vtx2morton = vec![0u32; num_vtx];
        let mut idx2vtx = vec![0u32; num_vtx];
        let mut idx2morton = vec![0u32; num_vtx];
        del_msh_cpu::mortons::sorted_morten_code3(
            &mut idx2vtx,
            &mut idx2morton,
            &mut vtx2morton,
            &vtx2xyz,
            &transform_world2unit,
        );
        let mut bnodes = vec![0u32; (num_vtx - 1) * 3];
        let mut bnode2depth = vec![0u32; (num_vtx - 1) * 3];
        del_msh_cpu::quad_oct_tree::binary_radix_tree_and_depth(
            &idx2morton,
            3,
            max_depth,
            &mut bnodes,
            &mut bnode2depth,
        );
        let mut bnode2onode = vec![0u32; num_vtx - 1];
        let mut idx2bnode = vec![0u32; num_vtx];
        del_msh_cpu::quad_oct_tree::bnode2onode_and_idx2bnode(
            &bnodes,
            &bnode2depth,
            &mut bnode2onode,
            &mut idx2bnode,
        );
        let num_onode = bnode2onode[num_vtx - 2] as usize + 1;
        let mut onodes = vec![u32::MAX; num_onode * 9];
        let mut onode2depth = vec![0u32; num_onode];
        let mut onode2center = vec![0f32; num_onode * 3];
        let mut idx2center = vec![0f32; num_vtx * 3];
        let mut idx2onode = vec![0u32; num_vtx];
        del_msh_cpu::quad_oct_tree::make_tree_from_binary_radix_tree(
            &bnodes,
            &bnode2onode,
            &bnode2depth,
            &idx2bnode,
            &idx2morton,
            num_onode,
            max_depth,
            3,
            &mut onodes,
            &mut onode2depth,
            &mut onode2center,
            &mut idx2onode,
            &mut idx2center,
        );
        del_msh_cpu::quad_oct_tree::check_octree::<3>(
            &idx2onode,
            &idx2center,
            &onodes,
            &onode2depth,
            &onode2center,
            max_depth,
        );
        del_msh_cpu::quad_oct_tree::check_octree_vtx2xyz::<3, 16>(
            &vtx2xyz,
            &transform_world2unit,
            &idx2vtx,
            &idx2onode,
            &idx2center,
            max_depth,
            &onode2center,
            &onode2depth,
        );
        let mut onode2nvtx = vec![0usize; num_onode];
        let mut onode2gcunit = vec![[0f32; 3]; num_onode];
        let mut onode2rhs = vec![[0f32; 3]; num_onode];
        for idx in 0..num_vtx {
            let i_vtx = idx2vtx[idx] as usize;
            let mut i_onode = idx2onode[idx] as usize;
            assert!(i_onode < num_onode);
            let pos_vtx_world = arrayref::array_ref![vtx2xyz, i_vtx * 3, 3];
            let pos_vtx_unit = del_geo_core::mat4_col_major::transform_homogeneous(
                &transform_world2unit,
                pos_vtx_world,
            )
            .unwrap();
            loop {
                onode2gcunit[i_onode][0] += pos_vtx_unit[0];
                onode2gcunit[i_onode][1] += pos_vtx_unit[1];
                onode2gcunit[i_onode][2] += pos_vtx_unit[2];
                onode2nvtx[i_onode] += 1;
                onode2rhs[i_onode][0] += vtx2rhs[i_vtx*3+0];
                onode2rhs[i_onode][1] += vtx2rhs[i_vtx*3+1];
                onode2rhs[i_onode][2] += vtx2rhs[i_vtx*3+2];
                if onodes[i_onode * 9] == u32::MAX {
                    break;
                }
                i_onode = onodes[i_onode * 9] as usize;
                assert!(i_onode < num_onode);
            }
        }
        for i_onode in 0..num_onode {
            assert!(onode2nvtx[i_onode] > 0);
            let s = 1.0 / onode2nvtx[i_onode] as f32;
            onode2gcunit[i_onode][0] *= s;
            onode2gcunit[i_onode][1] *= s;
            onode2gcunit[i_onode][2] *= s;
        }
        {
            // assertion
            let mut gc_world = [0f32; 3];
            vtx2xyz.chunks(3).for_each(|v| {
                gc_world[0] += v[0];
                gc_world[1] += v[1];
                gc_world[2] += v[2];
            });
            let s = 1.0 / num_vtx as f32;
            gc_world.iter_mut().for_each(|v| *v *= s);
            let gc_unit = del_geo_core::mat4_col_major::transform_homogeneous(
                &transform_world2unit,
                &gc_world,
            )
            .unwrap();
            use del_geo_core::vec3::Vec3;
            assert!(gc_unit.sub(&onode2gcunit[0]).norm() < 1.0e-6);
        }
        let mut vtx2lhs = vec![0f32; vtx2xyz.len()];
        del_msh_cpu::nbody::barnes_hut(
            &spoisson,
            &vtx2xyz,
            &vtx2rhs,
            &mut vtx2lhs,
            &transform_world2unit,
            del_msh_cpu::nbody::Octree {
                onodes: &onodes,
                onode2center: &onode2center,
                onode2depth: &onode2depth,
            },
            &idx2vtx,
            &onode2gcunit,
            &onode2rhs);
        vtx2lhs
    };
    hoge("target/green1b.obj".to_string(), &vtx2xyz, &vtx2lhs0);
    hoge("target/green1c.obj".to_string(), &vtx2xyz, &vtx2lhs1);

    del_msh_cpu::io_wavefront_obj::save_tri2vtx_vtx2xyz("target/green0.obj", &tri2vtx, &vtx2xyz, 3)
        .unwrap();
    //dbg!(tri2vtx);
}
