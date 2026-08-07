pub fn find_adjacent_edge_index(
    tv: &[usize; 3],
    ts: &[usize; 3],
    ied0: usize,
    tri2vtx: &[[usize; 3]],
) -> usize {
    let iv0 = tv[(ied0 + 1) % 3];
    let iv1 = tv[(ied0 + 2) % 3];
    assert_ne!(iv0, iv1);
    let it1 = ts[ied0];
    assert_ne!(it1, usize::MAX);
    if tri2vtx[it1][1] == iv1 && tri2vtx[it1][2] == iv0 {
        return 0;
    }
    if tri2vtx[it1][2] == iv1 && tri2vtx[it1][0] == iv0 {
        return 1;
    }
    if tri2vtx[it1][0] == iv1 && tri2vtx[it1][1] == iv0 {
        return 2;
    }
    panic!();
}

pub fn insert_a_point_inside_an_element(
    idx_vtx_insert: usize,
    idx_tri_insert: usize,
    tri2vtx: &mut Vec<[usize; 3]>,
    tri2tri: &mut Vec<[usize; 3]>,
    vtx2tri: &mut [usize],
) -> bool {
    assert_eq!(tri2vtx.len(), tri2tri.len());
    assert!(idx_tri_insert < tri2vtx.len());
    assert!(idx_vtx_insert < vtx2tri.len());

    let it_a = idx_tri_insert;
    let it_b = tri2vtx.len();
    let it_c = tri2vtx.len() + 1;

    tri2vtx.resize(tri2vtx.len() + 2, [usize::MAX; 3]);
    tri2tri.resize(tri2tri.len() + 2, [usize::MAX; 3]);
    let old_v: [usize; 3] = tri2vtx[idx_tri_insert];
    let old_s: [usize; 3] = tri2tri[idx_tri_insert];

    vtx2tri[idx_vtx_insert] = it_a;
    vtx2tri[old_v[0]] = it_b;
    vtx2tri[old_v[1]] = it_c;
    vtx2tri[old_v[2]] = it_a;

    tri2vtx[it_a] = [idx_vtx_insert, old_v[1], old_v[2]];
    tri2tri[it_a] = [old_s[0], it_b, it_c];
    if old_s[0] != usize::MAX {
        let jt0 = old_s[0];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_v, &old_s, 0, tri2vtx);
        tri2tri[jt0][jno0] = it_a;
    }

    tri2vtx[it_b] = [idx_vtx_insert, old_v[2], old_v[0]];
    tri2tri[it_b] = [old_s[1], it_c, it_a];
    if old_s[1] != usize::MAX {
        let jt0 = old_s[1];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_v, &old_s, 1, tri2vtx);
        tri2tri[jt0][jno0] = it_b;
    }

    tri2vtx[it_c] = [idx_vtx_insert, old_v[0], old_v[1]];
    tri2tri[it_c] = [old_s[2], it_a, it_b];
    if old_s[2] != usize::MAX {
        let jt0 = old_s[2];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_v, &old_s, 2, tri2vtx);
        tri2tri[jt0][jno0] = it_c;
    }
    true
}

pub fn insert_point_on_elem_edge(
    idx_vtx_insert: usize,
    idx_tri_insert: usize,
    idx_triedge_insert: usize,
    tri2vtx: &mut Vec<[usize; 3]>,
    tri2tri: &mut Vec<[usize; 3]>,
    vtx2tri: &mut [usize],
) -> bool {
    assert!(idx_tri_insert < tri2vtx.len());
    assert!(idx_vtx_insert < vtx2tri.len());
    assert_eq!(tri2vtx.len(), tri2tri.len());
    assert_ne!(tri2vtx[idx_tri_insert][idx_triedge_insert], usize::MAX);

    let itri_adj = tri2tri[idx_tri_insert][idx_triedge_insert];
    let ied_adj = find_adjacent_edge_index(
        &tri2vtx[idx_tri_insert],
        &tri2tri[idx_tri_insert],
        idx_triedge_insert,
        tri2vtx,
    );
    assert!(itri_adj < tri2vtx.len() && idx_triedge_insert < 3);

    let itri0 = idx_tri_insert;
    let itri1 = itri_adj;
    let itri2 = tri2vtx.len();
    let itri3 = tri2vtx.len() + 1;

    tri2vtx.resize(tri2vtx.len() + 2, [0; 3]);
    tri2tri.resize(tri2tri.len() + 2, [0; 3]);

    let old_a_v: [usize; 3] = tri2vtx[idx_tri_insert];
    let old_a_s: [usize; 3] = tri2tri[idx_tri_insert];
    let old_b_v: [usize; 3] = tri2vtx[itri_adj];
    let old_b_s: [usize; 3] = tri2tri[itri_adj];

    let ino_a0 = idx_triedge_insert;
    let ino_a1 = (idx_triedge_insert + 1) % 3;
    let ino_a2 = (idx_triedge_insert + 2) % 3;

    let ino_b0 = ied_adj;
    let ino_b1 = (ied_adj + 1) % 3;
    let ino_b2 = (ied_adj + 2) % 3;

    assert_eq!(old_a_v[ino_a1], old_b_v[ino_b2]);
    assert_eq!(old_a_v[ino_a2], old_b_v[ino_b1]);
    assert_eq!(old_a_s[ino_a0], itri1);
    assert_eq!(old_b_s[ino_b0], itri0);

    vtx2tri[idx_vtx_insert] = itri0;
    vtx2tri[old_a_v[ino_a2]] = itri0;
    vtx2tri[old_a_v[ino_a0]] = itri1;
    vtx2tri[old_b_v[ino_b2]] = itri2;
    vtx2tri[old_b_v[ino_b0]] = itri3;

    tri2vtx[itri0] = [idx_vtx_insert, old_a_v[ino_a2], old_a_v[ino_a0]];
    tri2tri[itri0] = [old_a_s[ino_a1], itri1, itri3];
    if old_a_s[ino_a1] != usize::MAX {
        let jt0 = old_a_s[ino_a1];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_a_v, &old_a_s, ino_a1, tri2vtx);
        tri2tri[jt0][jno0] = itri0;
    }

    tri2vtx[itri1] = [idx_vtx_insert, old_a_v[ino_a0], old_a_v[ino_a1]];
    tri2tri[itri1] = [old_a_s[ino_a2], itri2, itri0];
    if old_a_s[ino_a2] != usize::MAX {
        let jt0 = old_a_s[ino_a2];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_a_v, &old_a_s, ino_a2, tri2vtx);
        tri2tri[jt0][jno0] = itri1;
    }

    tri2vtx[itri2] = [idx_vtx_insert, old_b_v[ino_b2], old_b_v[ino_b0]];
    tri2tri[itri2] = [old_b_s[ino_b1], itri3, itri1];
    if old_b_s[ino_b1] != usize::MAX {
        let jt0 = old_b_s[ino_b1];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_b_v, &old_b_s, ino_b1, tri2vtx);
        tri2tri[jt0][jno0] = itri2;
    }

    tri2vtx[itri3] = [idx_vtx_insert, old_b_v[ino_b0], old_b_v[ino_b1]];
    tri2tri[itri3] = [old_b_s[ino_b2], itri0, itri2];
    if old_b_s[ino_b2] != usize::MAX {
        let jt0 = old_b_s[ino_b2];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_b_v, &old_b_s, ino_b2, tri2vtx);
        tri2tri[jt0][jno0] = itri3;
    }
    true
}

pub fn find_node(i_vtx: usize, tri2vtx: &[usize], i_tri: usize) -> usize {
    if tri2vtx[i_tri * 3] == i_vtx {
        return 0;
    }
    if tri2vtx[i_tri * 3 + 1] == i_vtx {
        return 1;
    }
    if tri2vtx[i_tri * 3 + 2] == i_vtx {
        return 2;
    }
    usize::MAX
}

pub fn flip_edge(
    itri_a: usize,
    ied0: usize,
    tri2vtx: &mut [[usize; 3]],
    tri2tri: &mut [[usize; 3]],
    vtx2tri: &mut [usize],
) -> bool {
    assert!(itri_a < tri2vtx.len() && ied0 < 3);
    if tri2tri[itri_a][ied0] == usize::MAX {
        return false;
    }

    let itri_b = tri2tri[itri_a][ied0];
    assert!(itri_b < tri2vtx.len());
    let ied1 = find_adjacent_edge_index(&tri2vtx[itri_a], &tri2tri[itri_a], ied0, tri2vtx);
    assert!(ied1 < 3);
    assert_eq!(tri2tri[itri_b][ied1], itri_a);

    let old_a_v: [usize; 3] = tri2vtx[itri_a];
    let old_a_s: [usize; 3] = tri2tri[itri_a];
    let old_b_v: [usize; 3] = tri2vtx[itri_b];
    let old_b_s: [usize; 3] = tri2tri[itri_b];

    let no_a0 = ied0;
    let no_a1 = (ied0 + 1) % 3;
    let no_a2 = (ied0 + 2) % 3;

    let no_b0 = ied1;
    let no_b1 = (ied1 + 1) % 3;
    let no_b2 = (ied1 + 2) % 3;

    assert_eq!(old_a_v[no_a1], old_b_v[no_b2]);
    assert_eq!(old_a_v[no_a2], old_b_v[no_b1]);

    vtx2tri[old_a_v[no_a1]] = itri_a;
    vtx2tri[old_a_v[no_a0]] = itri_a;
    vtx2tri[old_b_v[no_b1]] = itri_b;
    vtx2tri[old_b_v[no_b0]] = itri_b;

    tri2vtx[itri_a] = [old_a_v[no_a1], old_b_v[no_b0], old_a_v[no_a0]];
    tri2tri[itri_a] = [itri_b, old_a_s[no_a2], old_b_s[no_b1]];
    if old_a_s[no_a2] != usize::MAX {
        let jt0 = old_a_s[no_a2];
        assert!(jt0 < tri2vtx.len() && jt0 != itri_b && jt0 != itri_a);
        let jno0 = find_adjacent_edge_index(&old_a_v, &old_a_s, no_a2, tri2vtx);
        tri2tri[jt0][jno0] = itri_a;
    }
    if old_b_s[no_b1] != usize::MAX {
        let jt0 = old_b_s[no_b1];
        assert!(jt0 < tri2vtx.len() && jt0 != itri_b && jt0 != itri_a);
        let jno0 = find_adjacent_edge_index(&old_b_v, &old_b_s, no_b1, tri2vtx);
        tri2tri[jt0][jno0] = itri_a;
    }

    tri2vtx[itri_b] = [old_b_v[no_b1], old_a_v[no_a0], old_b_v[no_b0]];
    tri2tri[itri_b] = [itri_a, old_b_s[no_b2], old_a_s[no_a1]];
    if old_b_s[no_b2] != usize::MAX {
        let jt0 = old_b_s[no_b2];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_b_v, &old_b_s, no_b2, tri2vtx);
        tri2tri[jt0][jno0] = itri_b;
    }
    if old_a_s[no_a1] != usize::MAX {
        let jt0 = old_a_s[no_a1];
        assert!(jt0 < tri2vtx.len());
        let jno0 = find_adjacent_edge_index(&old_a_v, &old_a_s, no_a1, tri2vtx);
        tri2tri[jt0][jno0] = itri_b;
    }
    true
}

pub fn move_ccw(
    itri_cur: &mut usize,
    inotri_cur: &mut usize,
    itri_adj: usize,
    tri2vtx: &[[usize; 3]],
    tri2tri: &[[usize; 3]],
) -> bool {
    let inotri1 = (*inotri_cur + 1) % 3;
    if tri2tri[*itri_cur][inotri1] == itri_adj {
        return false;
    }
    let itri_nex = tri2tri[*itri_cur][inotri1];
    assert!(itri_nex < tri2vtx.len());
    let ino2 = find_adjacent_edge_index(&tri2vtx[*itri_cur], &tri2tri[*itri_cur], inotri1, tri2vtx);
    let inotri_nex = (ino2 + 1) % 3;
    assert_eq!(
        tri2vtx[*itri_cur][*inotri_cur],
        tri2vtx[itri_nex][inotri_nex]
    );
    *itri_cur = itri_nex;
    *inotri_cur = inotri_nex;
    true
}

pub fn move_cw(
    itri_cur: &mut usize,
    inotri_cur: &mut usize,
    itri_adj: usize,
    tri2vtx: &[[usize; 3]],
    tri2tri: &[[usize; 3]],
) -> bool {
    let inotri1 = (*inotri_cur + 2) % 3;
    if tri2tri[*itri_cur][inotri1] == itri_adj {
        return false;
    }
    let itri_nex = tri2tri[*itri_cur][inotri1];
    assert!(itri_nex < tri2vtx.len());
    let ino2 = find_adjacent_edge_index(&tri2vtx[*itri_cur], &tri2tri[*itri_cur], inotri1, tri2vtx);
    let inotri_nex = (ino2 + 2) % 3;
    assert_eq!(
        tri2vtx[*itri_cur][*inotri_cur],
        tri2vtx[itri_nex][inotri_nex]
    );
    *itri_cur = itri_nex;
    *inotri_cur = inotri_nex;
    true
}

pub fn find_edge_by_looking_around_point(
    ipo0: usize,
    ipo1: usize,
    tri2vtx: &[[usize; 3]],
    tri2tri: &[[usize; 3]],
    vtx2tri: &[usize],
) -> Option<(usize, usize, usize)> {
    let mut itc = vtx2tri[ipo0];
    let mut inc = find_node(ipo0, tri2vtx.as_flattened(), itc);
    loop {
        // serch clock-wise
        assert_eq!(tri2vtx[itc][inc], ipo0);
        let inotri2 = (inc + 1) % 3;
        if tri2vtx[itc][inotri2] == ipo1 {
            assert_eq!(tri2vtx[itc][inc], ipo0);
            assert_eq!(tri2vtx[itc][inotri2], ipo1);
            return Some((itc, inc, inotri2));
        }
        if !move_cw(&mut itc, &mut inc, usize::MAX, tri2vtx, tri2tri) {
            break;
        }
        if itc == vtx2tri[ipo0] {
            return None;
        }
    }
    // -------------
    itc = vtx2tri[ipo0];
    inc = find_node(ipo0, tri2vtx.as_flattened(), itc);
    loop {
        // search counter clock-wise
        assert_eq!(tri2vtx[itc][inc], ipo0);
        if !move_ccw(&mut itc, &mut inc, usize::MAX, tri2vtx, tri2tri) {
            break;
        }
        if itc == vtx2tri[ipo0] {
            // end if it goes around
            return None;
        }
        let inotri2 = (inc + 1) % 3;
        if tri2vtx[itc][inotri2] == ipo1 {
            assert_eq!(tri2vtx[itc][inc], ipo0);
            assert_eq!(tri2vtx[itc][inotri2], ipo1);
            return Some((itc, inc, inotri2));
        }
    }
    None
}

pub fn find_edge_by_looking_all_triangles(
    ipo0: usize,
    ipo1: usize,
    tri2vtx: &[[usize; 3]],
) -> Option<(usize, usize)> {
    for (itri, tri) in tri2vtx.iter().enumerate() {
        for iedtri in 0..3 {
            let jpo0 = tri[iedtri % 3];
            let jpo1 = tri[(iedtri + 1) % 3];
            if jpo0 == ipo0 && jpo1 == ipo1 {
                return Some((itri, iedtri));
            }
        }
    }
    None
}

pub fn flag_connected(tri2tri: &[[usize; 3]], itri0_ker: usize, iflag: i32) -> Vec<i32> {
    let mut inout_flg = vec![0; tri2tri.len()];
    assert!(itri0_ker < inout_flg.len());
    inout_flg[itri0_ker] = iflag;
    let mut ind_stack = Vec::<usize>::new();
    ind_stack.push(itri0_ker);
    loop {
        if ind_stack.is_empty() {
            break;
        }
        let itri_cur = ind_stack.pop().unwrap();
        for &jtri0 in &tri2tri[itri_cur] {
            if jtri0 == usize::MAX {
                continue;
            }
            if inout_flg[jtri0] != iflag {
                inout_flg[jtri0] = iflag;
                ind_stack.push(jtri0);
            }
        }
    }
    inout_flg
}

pub fn delete_tri_flag(
    tri2vtx0: &[[usize; 3]],
    tri2tri0: &[[usize; 3]],
    tri2flag0: &[i32],
    flag: i32,
) -> (Vec<[usize; 3]>, Vec<[usize; 3]>, Vec<i32>) {
    assert_eq!(tri2flag0.len(), tri2vtx0.len());
    let num_tri0 = tri2vtx0.len();
    let mut map01 = vec![usize::MAX; num_tri0];
    let mut num_tri1 = 0;
    for itri in 0..num_tri0 {
        if tri2flag0[itri] != flag {
            map01[itri] = num_tri1;
            num_tri1 += 1;
        }
    }
    let mut tri2vtx: Vec<[usize; 3]> = vec![[0; 3]; num_tri1];
    let mut tri2tri = vec![[0usize; 3]; num_tri1];
    let mut tri2flag = vec![-1; num_tri1];
    for itri0 in 0..tri2vtx0.len() {
        if map01[itri0] != usize::MAX {
            let itri1 = map01[itri0];
            assert!(itri1 < num_tri1);
            tri2vtx[itri1] = tri2vtx0[itri0];
            tri2tri[itri1] = tri2tri0[itri0];
            tri2flag[itri1] = tri2flag0[itri0];
            assert_ne!(tri2flag[itri1], flag);
        }
    }
    for face2tri in tri2tri.iter_mut() {
        for i_tri in face2tri.iter_mut() {
            if *i_tri == usize::MAX {
                continue;
            }
            let itri_s0 = *i_tri;
            assert!(itri_s0 < tri2vtx0.len());
            let jtri0 = map01[itri_s0];
            assert!(jtri0 < tri2vtx.len());
            *i_tri = jtri0;
        }
    }
    (tri2vtx, tri2tri, tri2flag)
}

/// Get the boundary edges' connectivity (bedge2vtx)
/// and connectivity between triangle and boundary edge (tri2tri)
///
/// # Return
/// (bedge2vtx, tri2tri)
pub fn boundaryedge2vtx(
    tri2vtx: &[[usize; 3]],
    num_vtx: usize,
) -> (Vec<[usize; 2]>, Vec<[usize; 3]>) {
    let num_tri = tri2vtx.len();
    let mut tri2tri: Vec<[usize; 3]> = crate::elem2elem::from_uniform_mesh(
        tri2vtx,
        &del_geo_core::tri::FACE2IDX,
        &del_geo_core::tri::IDX2NODE,
        num_vtx,
    )
    .as_chunks::<3>()
    .0
    .to_vec();
    let mut bedge2vtx: Vec<[usize; 2]> = vec![];
    for (i_tri, node2tri) in tri2tri.iter_mut().enumerate() {
        for i_node in 0..3 {
            if node2tri[i_node] != usize::MAX {
                continue;
            }
            node2tri[i_node] = num_tri + bedge2vtx.len();
            bedge2vtx.push([
                tri2vtx[i_tri][(i_node + 1) % 3],
                tri2vtx[i_tri][(i_node + 2) % 3],
            ]);
        }
    }
    (bedge2vtx, tri2tri)
}
