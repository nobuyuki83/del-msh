//! functions for polygon mesh

pub fn elem2area(elem2idx: &[usize], idx2vtx: &[usize], vtx2xy: &[f32]) -> Vec<f32> {
    let num_elem = elem2idx.len() - 1;
    let mut areas: Vec<f32> = vec![0f32; num_elem];
    for i_elem in 0..num_elem {
        let num_vtx_in_elem = elem2idx[i_elem + 1] - elem2idx[i_elem];
        for i_edge in 0..num_vtx_in_elem {
            let i0_vtx = idx2vtx[elem2idx[i_elem] + i_edge];
            let i1_vtx = idx2vtx[elem2idx[i_elem] + (i_edge + 1) % num_vtx_in_elem];
            areas[i_elem] += 0.5f32 * vtx2xy[i0_vtx * 2] * vtx2xy[i1_vtx * 2 + 1];
            areas[i_elem] -= 0.5f32 * vtx2xy[i0_vtx * 2 + 1] * vtx2xy[i1_vtx * 2];
        }
    }
    areas
}

pub fn vtx2vtx(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    num_vtx: usize,
    is_bidirectional: bool,
) -> (Vec<usize>, Vec<usize>) {
    let (vtx2jdx, jdx2elem) = crate::vtx2elem::from_polygon_mesh(elem2idx, idx2vtx, num_vtx);
    crate::vtx2vtx::from_polygon_mesh_edges_with_vtx2elem(
        elem2idx,
        idx2vtx,
        &vtx2jdx,
        &jdx2elem,
        is_bidirectional,
    )
}

pub fn vtx2elem(elem2idx: &[usize], idx2vtx: &[usize], num_vtx: usize) -> (Vec<usize>, Vec<usize>) {
    crate::vtx2elem::from_polygon_mesh(elem2idx, idx2vtx, num_vtx)
}

pub fn edge2vtx(elem2idx: &[usize], idx2vtx: &[usize], num_vtx: usize) -> Vec<usize> {
    crate::edge2vtx::from_polygon_mesh(elem2idx, idx2vtx, num_vtx)
}

pub fn elem2elem(elem2idx: &[usize], idx2vtx: &[usize], num_vtx: usize) -> Vec<usize> {
    crate::elem2elem::from_polygon_mesh(elem2idx, idx2vtx, num_vtx)
}
