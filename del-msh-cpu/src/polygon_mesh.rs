pub fn edge2vtx(elem2idx_offset: &[usize], idx2vtx: &[usize], num_vtx: usize) -> Vec<usize> {
    crate::edge2vtx::from_polygon_mesh(elem2idx_offset, idx2vtx, num_vtx)
}

pub fn elem2elem(elem2idx_offset: &[usize], idx2vtx: &[usize], num_vtx: usize) -> Vec<usize> {
    crate::elem2elem::from_polygon_mesh(elem2idx_offset, idx2vtx, num_vtx)
}
