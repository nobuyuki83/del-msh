//! methods that generate triangle mesh

/// split polygons of polygonal mesh into triangles
pub fn from_polygon_mesh(
    elem2idx: &[usize],
    idx2vtx: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut num_tri = 0_usize;
    for ielem in 0..elem2idx.len() - 1 {
        assert!(elem2idx[ielem + 1] >= elem2idx[ielem]);
        let num_node = elem2idx[ielem + 1] - elem2idx[ielem];
        num_tri += num_node - 2;
    }
    let mut tri2vtx = Vec::<usize>::with_capacity(num_tri * 3);
    let mut new2old = Vec::<usize>::with_capacity(num_tri);
    for ielem in 0..elem2idx.len() - 1 {
        let num_node = elem2idx[ielem + 1] - elem2idx[ielem];
        let idx0 = elem2idx[ielem];
        for i_node in 0..num_node - 2 {
            tri2vtx.push(idx2vtx[idx0 + 0]);
            tri2vtx.push(idx2vtx[idx0 + 1 + i_node]);
            tri2vtx.push(idx2vtx[idx0 + 2 + i_node]);
            new2old.push(ielem);
        }
    }
    (tri2vtx, new2old)
}

/// split quad element to triangle element
pub fn from_quad_mesh(
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

pub fn find_index_tri(
    tri2vtx: &[usize],
    i_vtx: usize) -> usize {
    if tri2vtx[0] == i_vtx { return 0; }
    if tri2vtx[1] == i_vtx { return 1; }
    if tri2vtx[2] == i_vtx { return 2; }
    panic!();
}