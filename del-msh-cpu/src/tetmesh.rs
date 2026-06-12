/// Given face `i_face` of tet `vtxs_i`, find the local face index on `j_tet` that shares the same
/// three vertices. Matching is done by vertex-index sum, so vertex order does not matter.
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

/// Logically delete `i_tet` from the mesh: severs all adjacency links to its neighbours in
/// `tet2tet` (both sides) and overwrites its vertex slots in `tet2vtx` with `usize::MAX`.
pub fn remove(i_tet: usize, tet2tet: &mut [usize], tet2vtx: &mut [usize]) {
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
            &del_geo_core::tet::FACE2IDX,
            &del_geo_core::tet::IDX2NODE,
        );
        tet2tet[i_tet * 4 + i2_node] = usize::MAX;
        tet2tet[k_tet * 4 + k2_node] = usize::MAX;
    }
    for i2_node in 0..4 {
        tet2vtx[i_tet * 4 + i2_node] = usize::MAX;
    }
}

/// Candidate for collapsing two adjacent boundary tets into a single pyramid.
/// `i0_node` is the boundary face node of `i_tet`; `i1_node` / `j0_node` identify the shared
/// interior apex vertex.
#[derive(Debug, Clone)]
struct MergeTwoTetsIntoPrm {
    i_tet: usize,
    i0_node: usize,
    i1_node: usize,
    j_tet: usize,
    j0_node: usize,
}

impl MergeTwoTetsIntoPrm {
    /// Returns the 5 vertex indices of the pyramid formed by merging the two tets.
    /// Winding: [base quad (k0,k3,k2,k1), apex k4].
    fn prism_vtxs(&self, tet2vtx: &[usize]) -> [usize; 5] {
        use del_geo_core::tet::{FACE2IDX, IDX2NODE};
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
            .find(|i| IDX2NODE[FACE2IDX[i0_node] + i] == i1_node)
            .unwrap();
        let i2_node = IDX2NODE[FACE2IDX[i0_node] + (i0_nofa + 1) % 3];
        let i3_node = IDX2NODE[FACE2IDX[i0_node] + (i0_nofa + 2) % 3];
        let j1_node = find_adjacent_face_index(
            arrayref::array_ref![tet2vtx, i_tet * 4, 4],
            i1_node,
            j_tet,
            &tet2vtx,
            &FACE2IDX,
            &IDX2NODE,
        );
        let k1_vtx = tet2vtx[i_tet * 4 + i2_node];
        let k2_vtx = tet2vtx[j_tet * 4 + j1_node];
        let k3_vtx = tet2vtx[i_tet * 4 + i3_node];
        [k0_vtx, k3_vtx, k2_vtx, k1_vtx, k4_vtx]
    }
}

/// Scan boundary tet pairs and greedily merge those whose resulting pyramid has a better
/// condition number (>1.3×) than both source tets. Returns `(pyrmd2vtx, consumed_tets)` where
/// `pyrmd2vtx` is a flat list of 5-vertex pyramids and `consumed_tets` is the set of tet indices
/// that were absorbed. Each tet is used at most once.
pub fn make_pyramids_on_boundary(
    tet2vtx: &[usize],
    tet2tet: &[usize],
    vtx2xyz: &[f64],
) -> (Vec<usize>, Vec<usize>) {
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
                let pnode2vtx = cand.prism_vtxs(tet2vtx);
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
        let pnode2vtx = cand.prism_vtxs(tet2vtx);
        pyrmd2vtx.extend_from_slice(&pnode2vtx);
        tets.insert(i_tet);
        tets.insert(j_tet);
    }
    (pyrmd2vtx, tets.into_iter().collect())
}
