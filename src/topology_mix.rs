//! query for topoologies of a mesh where element types are mixed

/// element surrounding point (elsup)
pub fn elsup(
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
        for ivtx0 in &idx2vtx[elem2idx[ielem]..elem2idx[ielem + 1]] {
            let jdx0 = vtx2jdx[*ivtx0];
            jdx2elem[jdx0] = ielem;
            vtx2jdx[*ivtx0] += 1;
        }
    }
    for ivtx in (1..num_vtx).rev() {
        vtx2jdx[ivtx] = vtx2jdx[ivtx - 1];
    }
    vtx2jdx[0] = 0;
    (vtx2jdx, jdx2elem)
}

#[test]
fn test_psup_mshmix() {
    let elem2idx = vec![0, 3, 7, 11, 14];
    let idx2vtx = vec![0, 4, 2, 4, 3, 5, 2, 1, 6, 7, 5, 3, 1, 5];
    let (vtx2jdx, jdx2elem) = elsup(&elem2idx, &idx2vtx, 8);
    assert_eq!(vtx2jdx, vec![0, 1, 3, 5, 7, 9, 12, 13, 14]);
    assert_eq!(jdx2elem, vec![0, 2, 3, 0, 1, 1, 3, 0, 1, 1, 2, 3, 2, 2]);
}


pub fn psupedge_from_meshtriquad(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    vtx2jdx: &[usize],
    jdx2elem: &[usize],
    is_bidirectional: bool) -> (Vec<usize>, Vec<usize>) {
    let nvtx = vtx2jdx.len() - 1;
    const EDGES_PAR_TRI: [usize; 6] = [0, 1, 1, 2, 2, 0];
    const EDGES_PAR_QUAD: [usize; 8] = [0, 1, 1, 2, 2, 3, 3, 0];

    let mut vtx2kdx = vec![0; nvtx + 1];
    let mut kdx2vtx = Vec::<usize>::new();

    for i_vtx in 0..nvtx {
        let mut set_vtx_idx = std::collections::BTreeSet::new();
        for ielem0 in &jdx2elem[vtx2jdx[i_vtx]..vtx2jdx[i_vtx + 1]] {
            let num_node = elem2idx[*ielem0 + 1] - elem2idx[*ielem0];
            let nedge = if num_node == 3 { 3 } else { 4 };
            for iedge in 0..nedge {
                let inode0: usize;
                let inode1: usize;
                {
                    if num_node == 3 {
                        inode0 = EDGES_PAR_TRI[iedge * 2 + 0];
                        inode1 = EDGES_PAR_TRI[iedge * 2 + 1];
                    } else {
                        inode0 = EDGES_PAR_QUAD[iedge * 2 + 0];
                        inode1 = EDGES_PAR_QUAD[iedge * 2 + 1];
                    }
                }
                let j_vtx0 = idx2vtx[elem2idx[*ielem0] + inode0];
                let j_vtx1 = idx2vtx[elem2idx[*ielem0] + inode1];
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
            kdx2vtx.push((*itr).try_into().unwrap());
        }
        vtx2kdx[i_vtx + 1] = vtx2kdx[i_vtx] + set_vtx_idx.len();
    }
    (vtx2kdx, kdx2vtx)
}

pub fn meshtri_from_meshtriquad(
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

pub fn meshline_from_meshtriquad(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    num_vtx: usize) -> Vec<usize> {
    use crate::topology_uniform::mshline_psup;
    let vtx2elem = elsup(
        &elem2idx, &idx2vtx,
        num_vtx);
    let vtx2vtx = psupedge_from_meshtriquad(
        &elem2idx, &idx2vtx,
        &vtx2elem.0, &vtx2elem.1,
        false);
     mshline_psup(&vtx2vtx.0, &vtx2vtx.1)
}