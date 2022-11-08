//! utility library to find topology information of uniform mesh
//! uniform mesh is a mesh that has single type of element (quad, tri, tet)

pub fn find_index_tri(
    tri_vtx: &[usize],
    ixyz: usize) -> usize {
    if tri_vtx[0] == ixyz { return 0; }
    if tri_vtx[1] == ixyz { return 1; }
    if tri_vtx[2] == ixyz { return 2; }
    panic!();
}

/// element surrounding points
pub fn elsup(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize) -> (Vec<usize>, Vec<usize>) {
    let num_elem = elem2vtx.len() / num_node;
    assert_eq!(elem2vtx.len(), num_elem * num_node);
    let mut vtx2idx = vec!(0_usize; num_vtx + 1);
    for i_elem in 0..num_elem {
        for i_node in 0..num_node {
            let i_vtx = elem2vtx[i_elem * num_node + i_node];
            assert!(i_vtx < num_vtx);
            vtx2idx[i_vtx + 1] += 1;
        }
    }
    for i_vtx in 0..num_vtx {
        vtx2idx[i_vtx + 1] += vtx2idx[i_vtx];
    }
    let num_vtx2elem = vtx2idx[num_vtx];
    let mut idx2elem = vec!(0; num_vtx2elem);
    for i_elem in 0..num_elem {
        for i_node in 0..num_node {
            let i_vtx0 = elem2vtx[i_elem * num_node + i_node];
            let iv2e = vtx2idx[i_vtx0];
            idx2elem[iv2e] = i_elem;
            vtx2idx[i_vtx0] += 1;
        }
    }
    for i_vtx in (1..num_vtx).rev() {
        vtx2idx[i_vtx] = vtx2idx[i_vtx - 1];
    }
    vtx2idx[0] = 0;
    (vtx2idx, idx2elem)
}

/// point surrounding point for mesh
/// * `elem2vtx` - map element to vertex: list of vertex index for each element
/// * `num_node` - number of vertex par elemnent (e.g., 3 for tri, 4 for quad)
/// * `num_vtx` - number of vertex
/// * `vtx2elem_idx` - map vertex to element index: cumulative sum
/// * `vtx2elem` - map vertex to element value: list of value
pub fn psup(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize,
    vtx2idx: &[usize],
    idx2elem: &[usize]) -> (Vec<usize>, Vec<usize>) {
    assert_eq!(vtx2idx.len(), num_vtx + 1);
    assert_eq!(elem2vtx.len() % num_node, 0);
    let mut vtx2flg = vec!(usize::MAX; num_vtx);
    let mut vtx2jdx = vec!(0_usize; num_vtx + 1);
    for i_vtx in 0..num_vtx {
        vtx2flg[i_vtx] = i_vtx;
        for idx0 in vtx2idx[i_vtx]..vtx2idx[i_vtx + 1] {
            let j_elem = idx2elem[idx0];
            for j_node in 0..num_node {
                let j_vtx = elem2vtx[j_elem * num_node + j_node];
                if vtx2flg[j_vtx] != i_vtx {
                    vtx2flg[j_vtx] = i_vtx;
                    vtx2jdx[i_vtx + 1] += 1;
                }
            }
        }
    }
    for i_vtx in 0..num_vtx {
        vtx2jdx[i_vtx + 1] += vtx2jdx[i_vtx];
    }
    let num_vtx2vtx = vtx2jdx[num_vtx];
    let mut jdx2vtx = vec!(0_usize; num_vtx2vtx);
    vtx2flg.iter_mut().for_each(|v| *v = usize::MAX );
    for i_vtx in 0..num_vtx {
        vtx2flg[i_vtx] = i_vtx;
        for idx0 in vtx2idx[i_vtx]..vtx2idx[i_vtx + 1] {
            let j_elem = idx2elem[idx0];
            for j_node in 0..num_node {
                let j_vtx = elem2vtx[j_elem * num_node + j_node];
                if vtx2flg[j_vtx] != i_vtx {
                    vtx2flg[j_vtx] = i_vtx;
                    let iv2v = vtx2jdx[i_vtx];
                    jdx2vtx[iv2v] = j_vtx;
                    vtx2jdx[i_vtx] += 1;
                }
            }
        }
    }
    for i_vtx in (1..num_vtx).rev() {
        vtx2jdx[i_vtx] = vtx2jdx[i_vtx - 1];
    }
    vtx2jdx[0] = 0;
    (vtx2jdx, jdx2vtx)
}

pub fn psup2(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize) -> (Vec<usize>, Vec<usize>)
{  // set pattern to sparse matrix
    assert_eq!(elem2vtx.len() % num_node, 0);
    let vtx2elem = elsup(
        &elem2vtx, num_node, num_vtx);
    assert_eq!(vtx2elem.0.len(), num_vtx + 1);
    let vtx2vtx = psup(
        &elem2vtx,
        num_node, num_vtx,
        &vtx2elem.0, &vtx2elem.1);
    assert_eq!(vtx2vtx.0.len(), num_vtx + 1);
    vtx2vtx
}

/// element surrounding element
/// * `elem2vtx` - vertex index of elements
/// * `num_node` - number of nodes par element
/// * `vtx2elem_idx` - jagged array index of element surrounding point
/// * `vtx2elem` - jagged array value of  element surrounding point
///
///  triangle: `face2jdx` = \[0,2,4,6]; `jdx2node` = \[1,2,2,0,0,1];
pub fn elsuel(
    elem2vtx: &[usize],
    num_node: usize,
    vtx2idx: &[usize],
    idx2elem: &[usize],
    face2jdx: &[usize],
    jdx2node: &[usize]) -> Vec<usize> {
    assert!(!vtx2idx.is_empty());
    let num_vtx = vtx2idx.len() - 1;
    let num_face_par_elem = face2jdx.len() - 1;
    let num_max_node_on_face = {
        let mut n0 = 0_usize;
        for i_face in 0..num_face_par_elem {
            let nno = face2jdx[i_face + 1] - face2jdx[i_face];
            n0 = if nno > n0 { nno } else { n0 }
        }
        n0
    };

    let num_elem = elem2vtx.len() / num_node;
    let mut elem2elem = vec!(usize::MAX; num_elem * num_face_par_elem);

    let mut vtx2flag = vec!(0; num_vtx); // vertex index -> flag
    let mut jdx2vtx = vec!(0; num_max_node_on_face);  // face node index -> vertex index
    for i_elem in 0..num_elem {
        for i_face in 0..num_face_par_elem {
            for jdx0 in 0..face2jdx[i_face + 1] - face2jdx[i_face] {
                let i_node0 = jdx2node[jdx0 + face2jdx[i_face]];
                assert!(i_node0 < num_node);
                let i_vtx = elem2vtx[i_elem * num_node + i_node0];
                assert!(i_vtx < num_vtx);
                jdx2vtx[jdx0] = i_vtx;
                vtx2flag[i_vtx] = 1;
            }
            let i_vtx0 = jdx2vtx[0];
            let mut flag0 = false;
            for idx0 in vtx2idx[i_vtx0]..vtx2idx[i_vtx0 + 1] {
                let j_elem0 = idx2elem[idx0];
                if j_elem0 == i_elem {
                    continue;
                }
                for j_face in 0..num_face_par_elem {
                    flag0 = true;
                    for jdx0 in face2jdx[j_face]..face2jdx[j_face + 1] {
                        let j_node0 = jdx2node[jdx0];
                        let j_vtx0 = elem2vtx[j_elem0 * num_node + j_node0];
                        if vtx2flag[j_vtx0] == 0 {
                            flag0 = false;
                            break;
                        }
                    }
                    if flag0 {
                        elem2elem[i_elem * num_face_par_elem + i_face] = j_elem0;
                        break;
                    }
                }
                if flag0 {
                    break;
                }
            }
            if !flag0 {
                elem2elem[i_elem * num_face_par_elem + i_face] = usize::MAX;
            }
            for ifano in 0..face2jdx[i_face + 1] - face2jdx[i_face] {
                vtx2flag[jdx2vtx[ifano]] = 0;
            }
        }
    }
    elem2elem
}

/// element surrounding element
/// * `elem2vtx` - vertex index of elements
/// * `num_node` - number of nodes par element
/// * `num_vtx` - number of vertices
///
///  triangle: face2idx = \[0,2,4,6]; idx2node = \[1,2,2,0,0,1];
pub fn elsuel2(
    elem2vtx: &[usize],
    num_node: usize,
    face2idx: &[usize],
    idx2node: &[usize],
    num_vtx: usize) -> Vec<usize> {
    let vtx2elem = elsup(
        &elem2vtx, num_node,
        num_vtx);
    elsuel(
        &elem2vtx, num_node,
        &vtx2elem.0, &vtx2elem.1,
        face2idx, idx2node)
}

// ------------------------------

pub fn psup_elem_edge(
    elem2vtx: &[usize],
    num_node: usize,
    edge2node: &[usize],
    vtx2idx: &Vec<usize>,
    idx2elem: &Vec<usize>,
    is_bidirectional: bool) -> (Vec<usize>, Vec<usize>) {
    let num_edge = edge2node.len() / 2;
    assert_eq!(edge2node.len(), num_edge * 2);

    let num_vtx = vtx2idx.len() - 1;
    let mut vtx2jdx = vec!(0_usize; num_vtx + 1);
    vtx2jdx[0] = 0;
    let mut jdx2vtx = Vec::<usize>::new();
    let mut set_vtx = std::collections::BTreeSet::new();
    for i_vtx in 0..num_vtx {
        set_vtx.clear();
        for idx0 in vtx2idx[i_vtx]..vtx2idx[i_vtx + 1] {
            let ielem0 = idx2elem[idx0];
            for iedge in 0..num_edge {
                let inode0 = edge2node[iedge * 2 + 0];
                let inode1 = edge2node[iedge * 2 + 1];
                let ivtx0 = elem2vtx[ielem0 * num_node + inode0];
                let ivtx1 = elem2vtx[ielem0 * num_node + inode1];
                if ivtx0 != i_vtx && ivtx1 != i_vtx { continue; }
                if ivtx0 == i_vtx {
                    if is_bidirectional || ivtx1 > i_vtx {
                        set_vtx.insert(ivtx1);
                    }
                } else {
                    if is_bidirectional || ivtx0 > i_vtx {
                        set_vtx.insert(ivtx0);
                    }
                }
            }
        }
        for vtx in &set_vtx {
            jdx2vtx.push((*vtx).try_into().unwrap());
        }
        vtx2jdx[i_vtx + 1] = vtx2jdx[i_vtx] + set_vtx.len();
    }
    (vtx2jdx, jdx2vtx)
}

/// making vertex indexes list of edges from psup (point surrounding point)
pub fn mshline_psup(
    vtx2idx: &Vec<usize>,
    idx2vtx: &Vec<usize>) -> Vec<usize> {
    let mut line2vtx = Vec::<usize>::with_capacity(idx2vtx.len() * 2);
    let num_vtx = vtx2idx.len() - 1;
    for i_vtx in 0..num_vtx {
        for idx0 in vtx2idx[i_vtx]..vtx2idx[i_vtx + 1] {
            let j_vtx = idx2vtx[idx0];
            line2vtx.push(i_vtx);
            line2vtx.push(j_vtx);
        }
    }
    line2vtx
}

pub fn mshline(
    elem2vtx: &[usize],
    num_node: usize,
    edge2node: &[usize],
    num_vtx: usize) -> Vec<usize>
{
    let vtx2elem = elsup(
        elem2vtx, num_node, num_vtx);
    let vtx2vtx = psup_elem_edge(
        elem2vtx, num_node,
        edge2node,
        &vtx2elem.0, &vtx2elem.1,
        false);
    mshline_psup(&vtx2vtx.0, &vtx2vtx.1)
}

pub fn tri_from_quad(
    quad2vtx: &[usize]) -> Vec<usize>
{
    let nquad = quad2vtx.len() / 4;
    let mut tri2vtx = vec![0; nquad * 2 * 3];
    for iquad in 0..nquad {
        tri2vtx[iquad * 6 + 0] = quad2vtx[iquad * 4 + 0];
        tri2vtx[iquad * 6 + 1] = quad2vtx[iquad * 4 + 1];
        tri2vtx[iquad * 6 + 2] = quad2vtx[iquad * 4 + 2];
        //
        tri2vtx[iquad * 6 + 3] = quad2vtx[iquad * 4 + 0];
        tri2vtx[iquad * 6 + 4] = quad2vtx[iquad * 4 + 2];
        tri2vtx[iquad * 6 + 5] = quad2vtx[iquad * 4 + 3];
    }
    tri2vtx
}

pub fn unify_separate_trimesh_indexing_xyz_uv(
    vtx_xyz2xyz: &Vec<f32>,
    vtx_uv2uv: &Vec<f32>,
    tri2vtx_xyz: &Vec<usize>,
    tri2vtx_uv: &Vec<usize>) -> (Vec<f32>, Vec<f32>, Vec<usize>, Vec<usize>, Vec<usize>)
{
    assert_eq!(tri2vtx_xyz.len(), tri2vtx_uv.len());
    let vtx_xyz2tri = elsup(
        tri2vtx_xyz, 3, vtx_xyz2xyz.len() / 3);
    let vtx_uv2tri = elsup(
        tri2vtx_uv, 3, vtx_uv2uv.len() / 2);
    let num_tri = tri2vtx_xyz.len() / 3;
    let mut tri2uni = vec!(usize::MAX; num_tri * 3);
    let mut uni2vtx_uv = Vec::<usize>::new();
    let mut uni2vtx_xyz = Vec::<usize>::new();
    for itri in 0..num_tri {
        for i_node in 0..3 {
            if tri2uni[itri * 3 + i_node] != usize::MAX { continue; }
            let ivtx_xyz = tri2vtx_xyz[itri * 3 + i_node];
            let ivtx_uv = tri2vtx_uv[itri * 3 + i_node];
            let s0 = &vtx_xyz2tri.1[vtx_xyz2tri.0[ivtx_xyz]..vtx_xyz2tri.0[ivtx_xyz + 1]];
            let s1 = &vtx_uv2tri.1[vtx_uv2tri.0[ivtx_uv]..vtx_uv2tri.0[ivtx_uv + 1]];
            let s0 = std::collections::BTreeSet::<&usize>::from_iter(s0.iter());
            let s1 = std::collections::BTreeSet::<&usize>::from_iter(s1.iter());
            let intersection: Vec<_> = s0.intersection(&s1).collect();
            if intersection.is_empty() { continue; }
            let iuni = uni2vtx_uv.len();
            uni2vtx_xyz.push(ivtx_xyz);
            uni2vtx_uv.push(ivtx_uv);
            for jtri0 in intersection.into_iter() {
                let jtri = *jtri0;
                let jno = find_index_tri(&tri2vtx_xyz[jtri*3..jtri*3+3], ivtx_xyz);
                assert_eq!(tri2vtx_xyz[jtri * 3 + jno], ivtx_xyz);
                assert_eq!(tri2vtx_uv[jtri * 3 + jno], ivtx_uv);
                assert_eq!(tri2uni[jtri * 3 + jno], usize::MAX);
                tri2uni[jtri * 3 + jno] = iuni;
            }
        }
    }
    let num_uni = uni2vtx_xyz.len();
    let mut uni2xyz = vec!(0_f32; num_uni * 3);
    for i_uni in 0..num_uni {
        let ivtx_xyz = uni2vtx_xyz[i_uni];
        uni2xyz[i_uni * 3 + 0] = vtx_xyz2xyz[ivtx_xyz * 3 + 0];
        uni2xyz[i_uni * 3 + 1] = vtx_xyz2xyz[ivtx_xyz * 3 + 1];
        uni2xyz[i_uni * 3 + 2] = vtx_xyz2xyz[ivtx_xyz * 3 + 2];
    }
    let mut uni2uv = vec!(0_f32; num_uni * 2);
    for i_uni in 0..num_uni {
        let i_vtx_uv = uni2vtx_uv[i_uni];
        uni2uv[i_uni * 2 + 0] = vtx_uv2uv[i_vtx_uv * 2 + 0];
        uni2uv[i_uni * 2 + 1] = vtx_uv2uv[i_vtx_uv * 2 + 1];
    }
    (uni2xyz, uni2uv, tri2uni, uni2vtx_xyz, uni2vtx_uv)
}