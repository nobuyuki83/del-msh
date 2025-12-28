fn hoge(path: String, vtx2xyz: &[f32], vt2lhs0: &[f32]) {
    let num_vtx = vtx2xyz.len() / 3;
    assert_eq!(vt2lhs0.len(), num_vtx * 3);
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
    del_msh_cpu::io_wavefront_obj::save_edge2vtx_vtx2xyz(path, &edge2vtxe, &vtxe2xyz, 3).unwrap();
}

fn main() {
    let vtx2xyz = {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        let mut vtx2xyz: Vec<f32> = (0..1000)
            .flat_map(|_i| {
                [
                    3.0 * rng.random::<f32>() - 1.0,
                    5.0 * rng.random::<f32>() - 2.0,
                    2.0 * rng.random::<f32>() - 3.0,
                ]
            })
            .collect();
        for _iter in 0..500 {
            let i_vtx = rng.random_range(0..vtx2xyz.len() / 3);
            vtx2xyz.push(vtx2xyz[i_vtx * 3 + 0]);
            vtx2xyz.push(vtx2xyz[i_vtx * 3 + 1]);
            vtx2xyz.push(vtx2xyz[i_vtx * 3 + 2]);
        }
        vtx2xyz
    };
    let num_vtx = vtx2xyz.len() / 3;
    let vtx2rhs = {
        let mut grad0 = vec![0f32; vtx2xyz.len()];
        /*
        grad0[0] = 1.0;
        grad0[1] = 0.0;
        grad0[2] = 0.0;
         */
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        grad0
            .iter_mut()
            .for_each(|v| *v = rng.random::<f32>() * 2.0 - 1.0);
        grad0
    };
    let spoisson = del_msh_cpu::nbody::NBodyModel::screened_poisson(10.0, 1.0e-3);
    let vtx2lhs0 = {
        let mut vtx2lhs = vec![0f32; vtx2xyz.len()];
        del_msh_cpu::nbody::filter_brute_force(
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
        let transform_world2unit = {
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
        let mut jdx2vtx = vec![0u32; num_vtx];
        let mut jdx2morton = vec![0u32; num_vtx];
        del_msh_cpu::mortons::sorted_morten_code3(
            &mut jdx2vtx,
            &mut jdx2morton,
            &mut vtx2morton,
            &vtx2xyz,
            &transform_world2unit,
        );
        let (idx2morton, idx2jdx_offset) = {
            let mut jdx2idx = vec![0u32; num_vtx];
            del_msh_cpu::array1d::unique_for_sorted_array(&jdx2morton, &mut jdx2idx);
            let num_idx = *jdx2idx.last().unwrap() as usize + 1;
            let mut idx2jdx_offset = vec![0u32; num_idx + 1];
            del_msh_cpu::map_idx::inverse(&jdx2idx, &mut idx2jdx_offset);
            let idx2morton = {
                let mut idx2morton = vec![0u32; num_idx];
                for idx in 0..num_idx as usize {
                    let jdx = idx2jdx_offset[idx] as usize;
                    assert!(jdx < num_vtx as usize);
                    idx2morton[idx] = jdx2morton[jdx];
                }
                (idx2morton, idx2jdx_offset)
            };
            idx2morton
        };
        let (onode2idx_tree, idx2onode, onode2center, onode2depth) = {
            let (onode2idx_tree, onode2center, onode2depth, idx2onode, idx2center) =
                del_msh_cpu::quad_oct_tree::construct_octree(&idx2morton, 10);
            del_msh_cpu::quad_oct_tree::check_octree_vtx2xyz::<3, 16>(
                &vtx2xyz,
                &transform_world2unit,
                &idx2jdx_offset,
                &jdx2vtx,
                &idx2onode,
                &idx2center,
                max_depth,
                &onode2center,
                &onode2depth,
            );
            (onode2idx_tree, idx2onode, onode2center, onode2depth)
        };
        let num_onode = onode2idx_tree.len() / 9;
        let mut onode2gcunit = vec![[0f32; 3]; num_onode];
        del_msh_cpu::quad_oct_tree::onode2gcuint_for_octree(
            &idx2jdx_offset,
            &jdx2vtx,
            &idx2onode,
            &vtx2xyz,
            &transform_world2unit,
            &onode2idx_tree,
            &mut onode2gcunit,
        );
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
        let mut onode2rhs = vec![0f32; num_onode * 3];
        del_msh_cpu::quad_oct_tree::aggregate_with_map(
            &idx2jdx_offset,
            3,
            &jdx2vtx,
            &vtx2rhs,
            &idx2onode,
            9,
            &onode2idx_tree,
            &mut onode2rhs,
        );
        let mut vtx2lhs = vec![0f32; vtx2xyz.len()];
        del_msh_cpu::nbody::barnes_hut(
            &spoisson,
            &vtx2xyz,
            &vtx2rhs,
            &vtx2xyz,
            &mut vtx2lhs,
            &transform_world2unit,
            del_msh_cpu::nbody::Octree {
                onodes: &onode2idx_tree,
                onode2center: &onode2center,
                onode2depth: &onode2depth,
            },
            &idx2jdx_offset,
            &jdx2vtx,
            &onode2gcunit,
            &onode2rhs,
            0.3,
        );
        vtx2lhs
    };
    {
        let diff = vtx2lhs0
            .iter()
            .zip(vtx2lhs1.iter())
            .map(|(a, b)| (a - b).abs())
            .reduce(f32::max)
            .unwrap_or(0.0);
        let scale = vtx2lhs0.iter().map(|a| a.abs()).sum::<f32>() / vtx2lhs0.len() as f32;
        dbg!(diff, scale);
    }
    hoge("target/green1b.obj".to_string(), &vtx2xyz, &vtx2lhs0);
    hoge("target/green1c.obj".to_string(), &vtx2xyz, &vtx2lhs1);
    //dbg!(tri2vtx);
}
