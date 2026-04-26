

pub fn find_adjacent_face_index(
    vtxs_i: &[usize; 4],
    i_face: usize,
    j_tet: usize,
    tet2vtx: &[usize],
    face2idx: &[usize],
    idx2node: &[usize],
) -> usize {
    let iv0 = vtxs_i[idx2node[face2idx[i_face] + 0]];
    let iv1 = vtxs_i[idx2node[face2idx[i_face] + 1]];
    let iv2 = vtxs_i[idx2node[face2idx[i_face] + 2]];
    let i_sum = iv0 + iv1 + iv2;
    assert_ne!(iv0, iv1);
    assert_ne!(iv0, iv2);
    let vtxs_j = arrayref::array_ref![tet2vtx, j_tet * 4, 4];
    for j_face in 0..4 {
        let jv0 = vtxs_j[idx2node[face2idx[j_face] + 0]];
        let jv1 = vtxs_j[idx2node[face2idx[j_face] + 1]];
        let jv2 = vtxs_j[idx2node[face2idx[j_face] + 2]];
        let j_sum = jv0 + jv1 + jv2;
        if i_sum == j_sum {
            return j_face;
        }
    }
    dbg!(iv0, iv1, iv2, &vtxs_j);
    panic!();
}

fn remove(i_tet: usize, tet2tet: &mut [usize], tet2vtx: &mut [usize]) {
    for i2_node in 0..4 {
        let k_tet = tet2tet[i_tet * 4 + i2_node];
        if k_tet == usize::MAX {
            continue;
        }
        let k2_node = find_adjacent_face_index(
            arrayref::array_ref![tet2vtx, i_tet * 4, 4],
            i2_node,
            k_tet,
            &tet2vtx,
            &crate::elem2elem::TET_FACE2IDX,
            &crate::elem2elem::TET_IDX2NODE,
        );
        tet2tet[i_tet * 4 + i2_node] = usize::MAX;
        tet2tet[k_tet * 4 + k2_node] = usize::MAX;
    }
    for i2_node in 0..4 {
        tet2vtx[i_tet * 4 + i2_node] = usize::MAX;
    }
}

#[derive(Debug, Clone)]
struct MergeTwoTetsIntoPrm {
    i_tet: usize,
    i0_node: usize,
    i1_node: usize,
    j_tet: usize,
    j0_node: usize,
}

impl MergeTwoTetsIntoPrm {
    fn prism_vtxs(&self, tet2vtx: &[usize]) -> [usize; 5] {
        use crate::elem2elem::{TET_FACE2IDX, TET_IDX2NODE};
        let (i_tet, j_tet, i0_node, i1_node, j0_node) = (
            self.i_tet,
            self.j_tet,
            self.i0_node,
            self.i1_node,
            self.j0_node,
        );
        let k4_vtx = tet2vtx[i_tet * 4 + i0_node];
        assert_eq!(tet2vtx[j_tet * 4 + j0_node], k4_vtx);
        let k0_vtx = tet2vtx[i_tet * 4 + i1_node];
        let i0_nofa = (0..3)
            .find(|i| TET_IDX2NODE[TET_FACE2IDX[i0_node] + i] == i1_node)
            .unwrap();
        let i2_node = TET_IDX2NODE[TET_FACE2IDX[i0_node] + (i0_nofa + 1) % 3];
        let i3_node = TET_IDX2NODE[TET_FACE2IDX[i0_node] + (i0_nofa + 2) % 3];
        let j1_node = find_adjacent_face_index(
            arrayref::array_ref![tet2vtx, i_tet * 4, 4],
            i1_node,
            j_tet,
            &tet2vtx,
            &TET_FACE2IDX,
            &TET_IDX2NODE,
        );
        let k1_vtx = tet2vtx[i_tet * 4 + i2_node];
        let k2_vtx = tet2vtx[j_tet * 4 + j1_node];
        let k3_vtx = tet2vtx[i_tet * 4 + i3_node];
        [k0_vtx, k3_vtx, k2_vtx, k1_vtx, k4_vtx]
    }
}

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
        crate::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz, 3).unwrap();
        crate::io_vtk::write_vtk_cells(&mut file, crate::io_vtk::VtkElementType::TETRA, &tet2vtx)
            .unwrap();
    }

    let mut tet2tet = crate::uniform_mesh::elem2elem(
        &tet2vtx,
        4,
        &crate::elem2elem::TET_FACE2IDX,
        &crate::elem2elem::TET_IDX2NODE,
        vtx2xyz.len(),
    );

    let tri2vtx = crate::elem2elem::extract_boundary_mesh_for_uniform_mesh(
        &tet2vtx,
        4,
        &tet2tet,
        &crate::elem2elem::TET_FACE2IDX,
        &crate::elem2elem::TET_IDX2NODE,
    );

    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(
        "../target/tetmesh_boundary.obj",
        &tri2vtx,
        &vtx2xyz,
        3,
    )
    .unwrap();

    let num_tet = tet2vtx.len() / 4;
    let mut cands: Vec<MergeTwoTetsIntoPrm> = vec![];
    for i_tet in 0..num_tet {
        let cands_i: Vec<MergeTwoTetsIntoPrm> = (0..4)
            .flat_map(|i0_node| {
                if tet2tet[i_tet * 4 + i0_node] != usize::MAX {
                    return None;
                }
                let i0_vtx = tet2vtx[i_tet * 4 + i0_node];
                let res = (0..3)
                    .map(|i| {
                        let i1_node = (i0_node + i) % 4;
                        let j_tet = tet2tet[i_tet * 4 + i1_node];
                        if j_tet == usize::MAX {
                            return None;
                        }
                        let j0_node = (0..4).find(|&i| tet2vtx[j_tet * 4 + i] == i0_vtx).unwrap();
                        if tet2tet[j_tet * 4 + j0_node] != usize::MAX {
                            return None;
                        }
                        Some(MergeTwoTetsIntoPrm {
                            i_tet,
                            i0_node,
                            i1_node,
                            j_tet,
                            j0_node,
                        })
                    })
                    .flatten()
                    .collect::<Vec<MergeTwoTetsIntoPrm>>();
                Some(res)
            })
            .into_iter()
            .flatten()
            .collect();
        cands.extend_from_slice(&cands_i);
    }

    let cands: Vec<_> = cands
        .into_iter()
        .filter(|cand| {
            let ci = {
                let i_tet = cand.i_tet;
                let p0i = arrayref::array_ref![vtx2xyz, tet2vtx[i_tet * 4 + 0] * 3, 3];
                let p1i = arrayref::array_ref![vtx2xyz, tet2vtx[i_tet * 4 + 1] * 3, 3];
                let p2i = arrayref::array_ref![vtx2xyz, tet2vtx[i_tet * 4 + 2] * 3, 3];
                let p3i = arrayref::array_ref![vtx2xyz, tet2vtx[i_tet * 4 + 3] * 3, 3];
                1.0 / del_geo_core::tet::condition_number(p0i, p1i, p2i, p3i).unwrap()
            };
            let cj = {
                let j_tet = cand.j_tet;
                let p0j = arrayref::array_ref![vtx2xyz, tet2vtx[j_tet * 4 + 0] * 3, 3];
                let p1j = arrayref::array_ref![vtx2xyz, tet2vtx[j_tet * 4 + 1] * 3, 3];
                let p2j = arrayref::array_ref![vtx2xyz, tet2vtx[j_tet * 4 + 2] * 3, 3];
                let p3j = arrayref::array_ref![vtx2xyz, tet2vtx[j_tet * 4 + 3] * 3, 3];
                1.0 / del_geo_core::tet::condition_number(p0j, p1j, p2j, p3j).unwrap()
            };
            let c_new = {
                let pnode2vtx = cand.prism_vtxs(&tet2vtx);
                let p0 = arrayref::array_ref![vtx2xyz, pnode2vtx[0] * 3, 3];
                let p1 = arrayref::array_ref![vtx2xyz, pnode2vtx[1] * 3, 3];
                let p2 = arrayref::array_ref![vtx2xyz, pnode2vtx[2] * 3, 3];
                let p3 = arrayref::array_ref![vtx2xyz, pnode2vtx[3] * 3, 3];
                let p4 = arrayref::array_ref![vtx2xyz, pnode2vtx[4] * 3, 3];
                use del_geo_core::pyramid::jacobian_determinant_and_conditiona_number as djac_cnd;
                let (d0, c0) = djac_cnd(p0, p1, p2, p3, p4, &[0.5, 0.5, 0.5]).unwrap();
                let (d1, c1) = djac_cnd(p0, p1, p2, p3, p4, &[0.5, 0.5, 0.0]).unwrap();
                let (d2, c2) = djac_cnd(p0, p1, p2, p3, p4, &[0.0, 0.0, 0.0]).unwrap();
                let (d3, c3) = djac_cnd(p0, p1, p2, p3, p4, &[1.0, 0.0, 0.0]).unwrap();
                let (d4, c4) = djac_cnd(p0, p1, p2, p3, p4, &[1.0, 1.0, 0.0]).unwrap();
                let (d5, c5) = djac_cnd(p0, p1, p2, p3, p4, &[0.0, 1.0, 0.0]).unwrap();
                let (c0, c1, c2, c3, c4, c5) =
                    (1.0 / c0, 1.0 / c1, 1.0 / c2, 1.0 / c3, 1.0 / c4, 1.0 / c5);
                let dmin = [d0, d1, d2, d3, d4, d5]
                    .iter()
                    .copied()
                    .min_by(|x, y| x.total_cmp(y));
                dmin.filter(|&d| d > 0.0).and_then(|_| {
                    [c0, c1, c2, c3, c4, c5]
                        .iter()
                        .copied()
                        .min_by(|x, y| x.total_cmp(y))
                })
            };
            //
            let c_old = ci.min(cj);
            c_new.map_or(false, |c| c > c_old * 1.3)
        })
        .collect();
    dbg!(cands.len());

    let mut tets = std::collections::HashSet::<usize>::new();
    let mut pyrmd2vtx = vec![0usize; 0];
    for cand in &cands {
        let i_tet = cand.i_tet;
        let j_tet = cand.j_tet;
        if tets.contains(&i_tet) || tets.contains(&j_tet) {
            continue;
        }
        let pnode2vtx = cand.prism_vtxs(&tet2vtx);
        pyrmd2vtx.extend_from_slice(&pnode2vtx);
        tets.insert(i_tet);
        tets.insert(j_tet);
    }
    for i_tet in tets {
        remove(i_tet, &mut tet2tet, &mut tet2vtx);
    }
    let tet2vtx = crate::extract::from_uniform_mesh_lambda(&tet2vtx, 4, |i_tet| {
        tet2vtx[i_tet * 4] != usize::MAX
    });
    dbg!(tet2vtx.len() / 4);
    dbg!(pyrmd2vtx.len() / 5);
    let prism2vtx = vec![0usize; 0];

    { // convert tet/prism mixed mesh to polyhedral mesh
        let num_elem = tet2vtx.len() / 4 + pyrmd2vtx.len() / 5 + prism2vtx.len() / 6;
        let mut elem2idx_offset = vec![0usize; num_elem + 1];
        let mut idx2vtx = vec![0usize; tet2vtx.len() + pyrmd2vtx.len() + prism2vtx.len()];
        dbg!("num_idx", idx2vtx.len());
        crate::mixed_mesh::to_polyhedron_mesh(
            &tet2vtx,
            &pyrmd2vtx,
            &prism2vtx,
            &mut elem2idx_offset,
            &mut idx2vtx,
        );
        {
            let mut elem2volume = vec![0f64; num_elem];
            crate::polyhedron_mesh::elem2volume(
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
        let (vtx2kdx_offset, kdx2elem) =
            crate::vtx2elem::from_polygon_mesh(&elem2idx_offset, &idx2vtx, vtx2xyz.len() / 3);
        let (elem2jdx, jdx2elem) = crate::polyhedron_mesh::elem2elem_with_vtx2elem(
            &elem2idx_offset,
            &idx2vtx,
            &vtx2kdx_offset,
            &kdx2elem,
        );
        let num_outer_face = jdx2elem.iter().filter(|&i| *i == usize::MAX).count();
        assert_eq!(num_outer_face, tri2vtx.len()/3 - pyrmd2vtx.len() / 5);
    }

    {
        let mut file = std::fs::File::create("../target/tetmesh1.vtk").expect("file not found.");
        crate::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz, 3).unwrap();
        crate::io_vtk::write_vtk_cells_mix(&mut file, &tet2vtx, &pyrmd2vtx, &prism2vtx).unwrap();
    }

    Ok(())
}
