//! functions for computing topology attributes for polygon mesh
//! data structure is `elem2idx_offset` and `idx2vtx` and num_vtx

pub fn vtx2vtx(
    elem2idx_offset: &[usize],
    idx2vtx: &[usize],
    num_vtx: usize,
    is_bidirectional: bool,
) -> (Vec<usize>, Vec<usize>) {
    let (vtx2jdx_offset, jdx2elem) =
        crate::vtx2elem::from_polygon_mesh(elem2idx_offset, idx2vtx, num_vtx);
    crate::vtx2vtx::from_polygon_mesh_edges_with_vtx2elem(
        elem2idx_offset,
        idx2vtx,
        &vtx2jdx_offset,
        &jdx2elem,
        is_bidirectional,
    )
}

pub fn vtx2elem(
    elem2idx_offset: &[usize],
    idx2vtx: &[usize],
    num_vtx: usize,
) -> (Vec<usize>, Vec<usize>) {
    crate::vtx2elem::from_polygon_mesh(elem2idx_offset, idx2vtx, num_vtx)
}
