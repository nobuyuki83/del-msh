//! methods that generate the elements adjacent to an element


pub fn face2node_of_polygon_element(num_node: usize) -> (Vec<usize>, Vec<usize>)
{
    let mut face2idx = vec!(0; num_node + 1);
    let mut idx2node = vec!(0; num_node * 2);
    for iedge in 0..num_node {
        face2idx[iedge + 1] = (iedge + 1) * 2;
        idx2node[iedge * 2 + 0] = iedge;
        idx2node[iedge * 2 + 1] = (iedge + 1) % num_node;
    }
    (face2idx, idx2node)
}


pub fn face2node_of_simplex_element(num_node: usize) -> (Vec<usize>, Vec<usize>)
{
    let num_node_face = num_node - 1;
    let mut face2idx = vec!(0; num_node + 1);
    let mut idx2node = vec!(0; num_node * num_node_face);
    for iedge in 0..num_node {
        face2idx[iedge + 1] = (iedge + 1) * 2;
        let mut icnt = 0;
        for ino in 0..num_node {
            let ino1 = (iedge + ino) % num_node;
            if ino1 == iedge {
                continue;
            }
            idx2node[iedge * num_node_face + icnt] = ino1;
            icnt += 1;
        }
    }
    (face2idx, idx2node)
}


/// element adjacency of uniform mesh
/// * `elem2vtx` - vertex index of elements
/// * `num_node` - number of nodes par element
/// * `vtx2elem_idx` - jagged array index of element surrounding point
/// * `vtx2elem` - jagged array value of  element surrounding point
///
///  triangle: `face2jdx` = \[0,2,4,6]; `jdx2node` = \[1,2,2,0,0,1];
pub fn from_uniform_mesh_with_vtx2elem(
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
pub fn from_uniform_mesh(
    elem2vtx: &[usize],
    num_node: usize,
    face2idx: &[usize],
    idx2node: &[usize],
    num_vtx: usize) -> Vec<usize> {
    let vtx2elem = crate::vtx2elem::from_uniform_mesh(
        &elem2vtx, num_node,
        num_vtx);
    from_uniform_mesh_with_vtx2elem(
        &elem2vtx, num_node,
        &vtx2elem.0, &vtx2elem.1,
        face2idx, idx2node)
}