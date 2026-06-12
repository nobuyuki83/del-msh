#[cfg(test)]
mod tests {

    #[test]
    fn hoge() -> anyhow::Result<(), anyhow::Error> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("parent directory not found")
            .join("asset")
            .join("spot")
            .join("tet_mesh.npz");
        let file = std::fs::File::open(&path)?;
        let mut npz = ndarray_npy::NpzReader::new(file)?;
        let vtx2xyz: ndarray::Array2<f64> = npz.by_name("points.npy")?;
        let tet2vtx: ndarray::Array2<i32> = npz.by_name("tets.npy")?;
        println!("points shape = {:?}", vtx2xyz.dim()); // (N, 3)
        println!("tets shape   = {:?}", tet2vtx.dim()); // (M, 4)

        let (vtx2xyz, vtx2xyz_offset) = vtx2xyz.to_owned().into_raw_vec_and_offset();
        let (tet2vtx, tet2vtx_offset) = tet2vtx.to_owned().into_raw_vec_and_offset();
        assert_eq!(vtx2xyz_offset, Some(0));
        assert_eq!(tet2vtx_offset, Some(0));

        let mut tet2vtx = tet2vtx.iter().map(|v| *v as usize).collect::<Vec<_>>();
        {
            let mut file = std::fs::File::create("../target/tetmesh.vtk").expect("file not found.");
            del_msh_cpu::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz, 3).unwrap();
            del_msh_cpu::io_vtk::write_vtk_cells(
                &mut file,
                del_msh_cpu::io_vtk::VtkElementType::TETRA,
                &tet2vtx,
            )
            .unwrap();
        }

        let mut tet2tet = del_msh_cpu::uniform_mesh::elem2elem(
            &tet2vtx,
            4,
            &del_geo_core::tet::FACE2IDX,
            &del_geo_core::tet::IDX2NODE,
            vtx2xyz.len(),
        );

        let tri2vtx = del_msh_cpu::elem2elem::extract_boundary_mesh_for_uniform_mesh(
            &tet2vtx,
            4,
            &tet2tet,
            &del_geo_core::tet::FACE2IDX,
            &del_geo_core::tet::IDX2NODE,
        );
        del_msh_cpu::io_wavefront_obj::save_tri2vtx_vtx2xyz(
            "../target/tetmesh_boundary.obj",
            &tri2vtx,
            &vtx2xyz,
            3,
        )
        .unwrap();
        //
        let num_outer_face_before = tet2tet.iter().filter(|&i| *i == usize::MAX).count();
        let (tet2vtx, pyrmd2vtx) = {
            let (pyrmd2vtx, tets) =
                del_msh_cpu::tetmesh::make_pyramids_on_boundary(&tet2vtx, &tet2tet, &vtx2xyz);
            for i_tet in tets {
                del_msh_cpu::tetmesh::remove(i_tet, &mut tet2tet, &mut tet2vtx);
            }
            let tet2vtx = del_msh_cpu::extract::from_uniform_mesh_lambda(&tet2vtx, 4, |i_tet| {
                tet2vtx[i_tet * 4] != usize::MAX
            });
            dbg!(tet2vtx.len() / 4);
            dbg!(pyrmd2vtx.len() / 5);
            (tet2vtx, pyrmd2vtx)
        };
        let (oelem2ldx_offset, ldx2vtx) = {
            // convert tet/prism mixed mesh to polyhedral mesh
            let (elem2idx_offset, idx2vtx) = {
                let prism2vtx = vec![];
                let num_elem = tet2vtx.len() / 4 + pyrmd2vtx.len() / 5 + prism2vtx.len() / 6;
                let mut elem2idx_offset = vec![0usize; num_elem + 1];
                let mut idx2vtx = vec![0usize; tet2vtx.len() + pyrmd2vtx.len() + prism2vtx.len()];
                dbg!("num_idx", idx2vtx.len());
                del_msh_cpu::mixed_mesh::to_polyhedron_mesh(
                    &tet2vtx,
                    &pyrmd2vtx,
                    &prism2vtx,
                    &vec![],
                    &mut elem2idx_offset,
                    &mut idx2vtx,
                );
                (elem2idx_offset, idx2vtx)
            };
            for i_elem in 0..elem2idx_offset.len() - 1 {
                let vtxs = &idx2vtx[elem2idx_offset[i_elem]..elem2idx_offset[i_elem + 1]];
                vtxs.iter()
                    .for_each(|&i_vtx| assert!(i_vtx < vtx2xyz.len() / 3));
            }
            {
                let mut elem2volume = vec![0f64; elem2idx_offset.len() - 1];
                del_msh_cpu::polyhedron_mesh::elem2volume(
                    &elem2idx_offset,
                    &idx2vtx,
                    &vtx2xyz,
                    0,
                    &mut elem2volume,
                );
                let vmin = elem2volume
                    .into_iter()
                    .min_by(|x, y| x.total_cmp(y))
                    .unwrap();
                assert!(vmin > 0.0);
            }
            let (elem2jdx_offset, jdx2elem) = {
                let (vtx2kdx_offset, kdx2elem) = del_msh_cpu::vtx2elem::from_polygon_mesh(
                    &elem2idx_offset,
                    &idx2vtx,
                    vtx2xyz.len() / 3,
                );
                del_msh_cpu::polyhedron_mesh::elem2elem_with_vtx2elem(
                    &elem2idx_offset,
                    &idx2vtx,
                    &vtx2kdx_offset,
                    &kdx2elem,
                )
            };
            {
                let num_outer_face_after = jdx2elem.iter().filter(|&i| *i == usize::MAX).count();
                assert_eq!(
                    num_outer_face_after,
                    num_outer_face_before - pyrmd2vtx.len() / 5
                );
            }
            del_msh_cpu::polyhedron_mesh::extract_boundary(
                &elem2idx_offset,
                &idx2vtx,
                &elem2jdx_offset,
                &jdx2elem,
            )
        };
        del_msh_cpu::io_wavefront_obj::save_elem2idx_idx2vtx_vtx2xyz(
            "../target/tetmesh_to_polyhedron_surf.obj",
            &oelem2ldx_offset,
            &ldx2vtx,
            &vtx2xyz,
            3,
        )
        .unwrap();

        {
            let mut file =
                std::fs::File::create("../target/tetmesh1.vtk").expect("file not found.");
            del_msh_cpu::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz, 3).unwrap();
            del_msh_cpu::io_vtk::write_vtk_cells_mix(
                &mut file,
                &tet2vtx,
                &pyrmd2vtx,
                &vec![],
                &vec![],
            )
            .unwrap();
        }

        let (nvtx2xyz, vtx2nvtx) = {
            let ovtx2vtxs: std::collections::HashSet<usize> = tri2vtx.iter().map(|&i| i).collect();
            assert_eq!(
                ovtx2vtxs.len(),
                ldx2vtx
                    .iter()
                    .map(|&i| i)
                    .collect::<std::collections::HashSet<usize>>()
                    .len()
            );
            let vtx2ovtx: std::collections::HashMap<usize, usize> = ovtx2vtxs
                .iter()
                .enumerate()
                .map(|(i_ovtx, &i_vtx)| (i_vtx, i_ovtx))
                .collect();
            let num_vtx = vtx2xyz.len() / 3;
            let num_ovtx = ovtx2vtxs.len();
            let mut ovtx2nrm = vec![0f64; num_ovtx * 3];
            del_msh_cpu::trimesh3::vtx2normal_with_mapping(
                &tri2vtx,
                &vtx2xyz,
                &vtx2ovtx,
                &mut ovtx2nrm,
            );
            let mut vtx2nvtx = std::collections::HashMap::<usize, usize>::new();
            let mut nvtx2xyz: Vec<f64> = vtx2xyz.clone();
            for (i_ovtx, &i_vtx) in ovtx2vtxs.iter().enumerate() {
                let xyz0 = arrayref::array_ref![vtx2xyz, i_vtx * 3, 3];
                let nrm = arrayref::array_ref![ovtx2nrm, i_ovtx * 3, 3];
                let dxyz = del_geo_core::vec3::scale(nrm, 0.005);
                let xyz1 = del_geo_core::vec3::add(xyz0, &dxyz);
                //let xyz2 = crate::trimesh3::extend_avoid_intersection(&tri2vtx, &vtx2xyz, &xyz1, 0.002);
                nvtx2xyz.extend_from_slice(&xyz1);
                vtx2nvtx.insert(i_vtx, i_ovtx + num_vtx);
            }
            (nvtx2xyz, vtx2nvtx)
        };

        let mut hex2nvtx: Vec<usize> = vec![];
        let mut prism2nvtx: Vec<usize> = vec![];
        for i_oelem in 0..oelem2ldx_offset.len() - 1 {
            let node2vtx = &ldx2vtx[oelem2ldx_offset[i_oelem]..oelem2ldx_offset[i_oelem + 1]];
            match node2vtx.len() {
                3 => {
                    prism2nvtx.extend_from_slice(&[
                        node2vtx[0],
                        node2vtx[1],
                        node2vtx[2],
                        *vtx2nvtx.get(&node2vtx[0]).unwrap(),
                        *vtx2nvtx.get(&node2vtx[1]).unwrap(),
                        *vtx2nvtx.get(&node2vtx[2]).unwrap(),
                    ]);
                }
                4 => {
                    hex2nvtx.extend_from_slice(&[
                        node2vtx[0],
                        node2vtx[1],
                        node2vtx[2],
                        node2vtx[3],
                        *vtx2nvtx.get(&node2vtx[0]).unwrap(),
                        *vtx2nvtx.get(&node2vtx[1]).unwrap(),
                        *vtx2nvtx.get(&node2vtx[2]).unwrap(),
                        *vtx2nvtx.get(&node2vtx[3]).unwrap(),
                    ]);
                }
                _ => unreachable!(),
            };
        }
        println!("num_hex {}", hex2nvtx.len() / 8);
        println!("num_prism {}", prism2nvtx.len() / 6);
        {
            let mut file =
                std::fs::File::create("../target/tetmesh2.vtk").expect("file not found.");
            del_msh_cpu::io_vtk::write_vtk_points(&mut file, "hoge", &nvtx2xyz, 3).unwrap();
            del_msh_cpu::io_vtk::write_vtk_cells_mix(
                &mut file,
                &tet2vtx,
                &pyrmd2vtx,
                &prism2nvtx,
                &hex2nvtx,
            )
            .unwrap();
        }
        for i_prism in 0..prism2nvtx.len() / 6 {
            let p = |i: usize| arrayref::array_ref![nvtx2xyz, prism2nvtx[i_prism * 6 + i] * 3, 3];
            let vol = del_geo_core::prism::volume(p(0), p(1), p(2), p(3), p(4), p(5), 1);
            assert!(vol > 0.0, "prism {i_prism} has non-positive volume {vol}");
        }
        Ok(())
    }
}
