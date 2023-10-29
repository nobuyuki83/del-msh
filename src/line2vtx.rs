//! methods related to line mesh

/// making vertex indexes list of line mesh from vertex surrounding vertex
/// * `vtx2idx` - vertex to index list
/// * `idx2vtx` - index to vertex list
pub fn from_vtx2vtx(
    vtx2idx: &Vec<usize>,
    idx2vtx: &Vec<usize>) -> Vec<usize> {
    let mut line2vtx = Vec::<usize>::with_capacity(idx2vtx.len() * 2);
    for i_vtx in 0..vtx2idx.len() - 1 {
        for &j_vtx in &idx2vtx[vtx2idx[i_vtx]..vtx2idx[i_vtx + 1]] {
            line2vtx.push(i_vtx);
            line2vtx.push(j_vtx);
        }
    }
    line2vtx
}

pub fn from_sepecific_edges_of_uniform_mesh(
    elem2vtx: &[usize],
    num_node: usize,
    edge2node: &[usize],
    num_vtx: usize) -> Vec<usize>
{
    let vtx2elem = crate::vtx2elem::from_uniform_mesh(
        elem2vtx, num_node, num_vtx);
    let vtx2vtx = crate::vtx2vtx::from_specific_edges_of_uniform_mesh(
        elem2vtx, num_node,
        edge2node,
        &vtx2elem.0, &vtx2elem.1,
        false);
    from_vtx2vtx(&vtx2vtx.0, &vtx2vtx.1)
}

/// generate line mesh as edges of polygon mesh
/// polygon mesh is a mixture of triangle, quadrilateal, pentagon mesh
/// * `num_vtx` - number of vertex
pub fn edge_of_polygon_mesh(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    num_vtx: usize) -> Vec<usize> {
    let vtx2elem = crate::vtx2elem::from_polygon_mesh(
        &elem2idx, &idx2vtx,
        num_vtx);
    let vtx2vtx = crate::vtx2vtx::edges_of_polygon_mesh(
        &elem2idx, &idx2vtx,
        &vtx2elem.0, &vtx2elem.1,
        false);
    crate::line2vtx::from_vtx2vtx(&vtx2vtx.0, &vtx2vtx.1)
}