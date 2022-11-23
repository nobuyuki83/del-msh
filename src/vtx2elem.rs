//! method generate list of elemement indices surrounding a vertex

/// element surrounding point (elsup)
pub fn from_mix_mesh(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    num_vtx: usize) -> (Vec<usize>, Vec<usize>)
{
    let num_elem = elem2idx.len() - 1;
    let mut vtx2jdx = Vec::new();
    vtx2jdx.resize(num_vtx + 1, 0);
    for ielem in 0..num_elem {
        for ivtx0 in &idx2vtx[elem2idx[ielem]..elem2idx[ielem + 1]] {
            vtx2jdx[ivtx0 + 1] += 1;
        }
    }
    for ivtx in 0..num_vtx {
        vtx2jdx[ivtx + 1] += vtx2jdx[ivtx];
    }
    let num_jdx = vtx2jdx[num_vtx];
    let mut jdx2elem = vec!(0; num_jdx);
    for ielem in 0..num_elem {
        for &i_vtx0 in &idx2vtx[elem2idx[ielem]..elem2idx[ielem + 1]] {
            let jdx0 = vtx2jdx[i_vtx0];
            jdx2elem[jdx0] = ielem;
            vtx2jdx[i_vtx0] += 1;
        }
    }
    for ivtx in (1..num_vtx).rev() {
        vtx2jdx[ivtx] = vtx2jdx[ivtx - 1];
    }
    vtx2jdx[0] = 0;
    (vtx2jdx, jdx2elem)
}

#[test]
fn test_mix_mesh() {
    let elem2idx = vec![0, 3, 7, 11, 14];
    let idx2vtx = vec![0, 4, 2, 4, 3, 5, 2, 1, 6, 7, 5, 3, 1, 5];
    let (vtx2jdx, jdx2elem) = from_mix_mesh(&elem2idx, &idx2vtx, 8);
    assert_eq!(vtx2jdx, vec![0, 1, 3, 5, 7, 9, 12, 13, 14]);
    assert_eq!(jdx2elem, vec![0, 2, 3, 0, 1, 1, 3, 0, 1, 1, 2, 3, 2, 2]);
}


/// element surrounding points
pub fn from_uniform_mesh(
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