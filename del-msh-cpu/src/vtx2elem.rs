//! methods that generate list of elements surrounding a vertex

use num_traits::AsPrimitive;

/// elements surrounding a vertex.
/// The element can be mixture of line, polygon and polyhedron
pub fn from_polygon_mesh(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    num_vtx: usize,
) -> (Vec<usize>, Vec<usize>) {
    let num_elem = elem2idx.len() - 1;
    let mut vtx2jdx = vec![0usize; num_vtx + 1];
    for i_elem in 0..num_elem {
        for i0_vtx in &idx2vtx[elem2idx[i_elem]..elem2idx[i_elem + 1]] {
            vtx2jdx[i0_vtx + 1] += 1;
        }
    }
    for i_vtx in 0..num_vtx {
        vtx2jdx[i_vtx + 1] += vtx2jdx[i_vtx];
    }
    let num_jdx = vtx2jdx[num_vtx];
    let mut jdx2elem = vec![0; num_jdx];
    for i_elem in 0..num_elem {
        for &i_vtx0 in &idx2vtx[elem2idx[i_elem]..elem2idx[i_elem + 1]] {
            let jdx0 = vtx2jdx[i_vtx0];
            jdx2elem[jdx0] = i_elem;
            vtx2jdx[i_vtx0] += 1;
        }
    }
    for i_vtx in (1..num_vtx).rev() {
        vtx2jdx[i_vtx] = vtx2jdx[i_vtx - 1];
    }
    vtx2jdx[0] = 0;
    (vtx2jdx, jdx2elem)
}

#[test]
fn test_polygon_mesh() {
    let elem2idx = vec![0, 3, 7, 11, 14];
    let idx2vtx = vec![0, 4, 2, 4, 3, 5, 2, 1, 6, 7, 5, 3, 1, 5];
    let (vtx2jdx, jdx2elem) = from_polygon_mesh(&elem2idx, &idx2vtx, 8);
    assert_eq!(vtx2jdx, vec![0, 1, 3, 5, 7, 9, 12, 13, 14]);
    assert_eq!(jdx2elem, vec![0, 2, 3, 0, 1, 1, 3, 0, 1, 1, 2, 3, 2, 2]);
}

/// element surrounding vertex
pub fn from_uniform_mesh<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    num_vtx: usize,
) -> (Vec<Index>, Vec<Index>)
where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let num_elem = elem2vtx.len() / num_node;
    assert_eq!(elem2vtx.len(), num_elem * num_node);
    let mut vtx2idx = vec![Index::zero(); num_vtx + 1];
    for i_elem in 0..num_elem {
        for i_node in 0..num_node {
            let i_vtx: usize = elem2vtx[i_elem * num_node + i_node].as_();
            assert!(i_vtx < num_vtx);
            vtx2idx[i_vtx + 1] = vtx2idx[i_vtx + 1] + Index::one();
        }
    }
    for i_vtx in 0..num_vtx {
        let tmp = vtx2idx[i_vtx];
        vtx2idx[i_vtx + 1] = vtx2idx[i_vtx + 1] + tmp;
    }
    let num_vtx2elem: usize = vtx2idx[num_vtx].as_();
    let mut idx2elem = vec![Index::zero(); num_vtx2elem];
    for i_elem in 0..num_elem {
        for i_node in 0..num_node {
            let i_vtx0: usize = elem2vtx[i_elem * num_node + i_node].as_();
            let iv2e: usize = vtx2idx[i_vtx0].as_();
            idx2elem[iv2e] = i_elem.as_();
            vtx2idx[i_vtx0] = vtx2idx[i_vtx0] + Index::one();
        }
    }
    for i_vtx in (1..num_vtx).rev() {
        vtx2idx[i_vtx] = vtx2idx[i_vtx - 1];
    }
    vtx2idx[0] = Index::zero();
    (vtx2idx, idx2elem)
}
