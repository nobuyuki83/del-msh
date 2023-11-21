//! methods that generate vertices connected to a vertex

/// point surrounding point for mesh
/// * `elem2vtx` - map element to vertex: list of vertex index for each element
/// * `num_node` - number of vertex par elemnent (e.g., 3 for tri, 4 for quad)
/// * `num_vtx` - number of vertex
/// * `vtx2idx` - map vertex to element index: cumulative sum
/// * `idx2elem` - map vertex to element value: list of value
pub fn from_uniform_mesh(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize,
    vtx2idx: &[usize],
    idx2elem: &[usize]) -> (Vec<usize>, Vec<usize>)
{
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
    vtx2flg.iter_mut().for_each(|v| *v = usize::MAX);
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

pub fn from_uniform_mesh2(
    elem2vtx: &[usize],
    num_node: usize,
    num_vtx: usize) -> (Vec<usize>, Vec<usize>)
{  // set pattern to sparse matrix
    assert_eq!(elem2vtx.len() % num_node, 0);
    let vtx2elem = crate::vtx2elem::from_uniform_mesh(
        elem2vtx, num_node, num_vtx);
    assert_eq!(vtx2elem.0.len(), num_vtx + 1);
    let vtx2vtx = from_uniform_mesh(
        elem2vtx, num_node, num_vtx,
        &vtx2elem.0, &vtx2elem.1);
    assert_eq!(vtx2vtx.0.len(), num_vtx + 1);
    vtx2vtx
}

pub fn from_specific_edges_of_uniform_mesh(
    elem2vtx: &[usize],
    num_node: usize,
    edge2node: &[usize],
    vtx2idx: &Vec<usize>,
    idx2elem: &[usize],
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
            jdx2vtx.push(*vtx);
        }
        vtx2jdx[i_vtx + 1] = vtx2jdx[i_vtx] + set_vtx.len();
    }
    (vtx2jdx, jdx2vtx)
}


/// make vertex surrounding vertex as edges of polygon mesh.
/// A polygon mesh is a mixture of elements such as triangle, quadrilateal, pentagon.
pub fn edges_of_polygon_mesh(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    vtx2jdx: &[usize],
    jdx2elem: &[usize],
    is_bidirectional: bool) -> (Vec<usize>, Vec<usize>) {
    let nvtx = vtx2jdx.len() - 1;

    let mut vtx2kdx = vec![0; nvtx + 1];
    let mut kdx2vtx = Vec::<usize>::new();

    for i_vtx in 0..nvtx {
        let mut set_vtx_idx = std::collections::BTreeSet::new();
        for &ielem0 in &jdx2elem[vtx2jdx[i_vtx]..vtx2jdx[i_vtx + 1]] {
            let num_node = elem2idx[ielem0 + 1] - elem2idx[ielem0];
            let num_edge = num_node;
            for i_edge in 0..num_edge {
                let i_node0 = i_edge;
                let i_node1 = (i_edge + 1) % num_node;
                let j_vtx0 = idx2vtx[elem2idx[ielem0] + i_node0];
                let j_vtx1 = idx2vtx[elem2idx[ielem0] + i_node1];
                if j_vtx0 != i_vtx && j_vtx1 != i_vtx { continue; }
                if j_vtx0 == i_vtx {
                    if is_bidirectional || j_vtx1 > i_vtx {
                        set_vtx_idx.insert(j_vtx1);
                    }
                } else {
                    if is_bidirectional || j_vtx0 > i_vtx {
                        set_vtx_idx.insert(j_vtx0);
                    }
                }
            }
        }
        for itr in &set_vtx_idx {
            kdx2vtx.push(*itr);
        }
        vtx2kdx[i_vtx + 1] = vtx2kdx[i_vtx] + set_vtx_idx.len();
    }
    (vtx2kdx, kdx2vtx)
}

