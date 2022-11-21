//! methods that generate triangle mesh

/// split quad element into triangle element
pub fn from_tri_quad_mesh(
    elem2idx: &[usize],
    idx2vtx: &[usize]) -> Vec<usize> {
    let mut num_tri = 0_usize;
    for ielem in 0..elem2idx.len() - 1 {
        let nnode = elem2idx[ielem + 1] - elem2idx[ielem];
        if nnode == 3 { num_tri += 1; } else if nnode == 4 { num_tri += 2; }
    }
    let mut tri2vtx = Vec::<usize>::new();
    tri2vtx.reserve(num_tri * 3);
    for ielem in 0..elem2idx.len() - 1 {
        let nnode = elem2idx[ielem + 1] - elem2idx[ielem];
        let idx0 = elem2idx[ielem];
        if nnode == 3 {
            tri2vtx.push(idx2vtx[idx0 + 0]);
            tri2vtx.push(idx2vtx[idx0 + 1]);
            tri2vtx.push(idx2vtx[idx0 + 2]);
        } else if nnode == 4 {
            tri2vtx.push(idx2vtx[idx0 + 0]);
            tri2vtx.push(idx2vtx[idx0 + 1]);
            tri2vtx.push(idx2vtx[idx0 + 2]);
            //
            tri2vtx.push(idx2vtx[idx0 + 0]);
            tri2vtx.push(idx2vtx[idx0 + 2]);
            tri2vtx.push(idx2vtx[idx0 + 3]);
        }
    }
    tri2vtx
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