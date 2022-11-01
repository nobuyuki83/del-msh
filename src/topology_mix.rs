/// element surrounding point (elsup)
pub fn elsup(
    elem2vtx_idx: &[usize],
    elem2vtx: &[usize],
    num_vtx: usize) -> (Vec<usize>, Vec<usize>)
{
    let num_elem = elem2vtx_idx.len() - 1;
    let mut vtx2elem_idx = Vec::new();
    vtx2elem_idx.resize(num_vtx + 1, 0);
    for ielem in 0..num_elem {
        for iivtx in elem2vtx_idx[ielem]..elem2vtx_idx[ielem + 1] {
            let ivtx0 = elem2vtx[iivtx];
            let ivtx0: usize = ivtx0 as usize;
            vtx2elem_idx[ivtx0 + 1] += 1;
        }
    }
    for ivtx in 0..num_vtx {
        vtx2elem_idx[ivtx + 1] += vtx2elem_idx[ivtx];
    }
    let nvtx2elem = vtx2elem_idx[num_vtx];
    let mut vtx2elem = vec!(0; nvtx2elem);
    for ielem in 0..num_elem {
        for iivtx in elem2vtx_idx[ielem]..elem2vtx_idx[ielem + 1] {
            let ivtx0 = elem2vtx[iivtx];
            let ivtx0 = ivtx0 as usize;
            let ind1 = vtx2elem_idx[ivtx0];
            vtx2elem[ind1] = ielem;
            vtx2elem_idx[ivtx0] += 1;
        }
    }
    for ivtx in (1..num_vtx).rev() {
        vtx2elem_idx[ivtx] = vtx2elem_idx[ivtx - 1];
    }
    vtx2elem_idx[0] = 0;
    (vtx2elem_idx, vtx2elem)
}

#[test]
fn test_psup_mshmix() {
    let elem2vtx_idx = vec![0, 3, 7, 11, 14];
    let elem2vtx = vec![0, 4, 2, 4, 3, 5, 2, 1, 6, 7, 5, 3, 1, 5];
    let (vtx2elem_idx, vtx2elem) = elsup(&elem2vtx_idx, &elem2vtx, 8);
    assert_eq!(vtx2elem_idx, vec![0, 1, 3, 5, 7, 9, 12, 13, 14]);
    assert_eq!(vtx2elem, vec![0, 2, 3, 0, 1, 1, 3, 0, 1, 1, 2, 3, 2, 2]);
}


pub fn psupedge_from_meshtriquad(
    elem2vtx_index: &[usize],
    elem2vtx: &[usize],
    vtx2elem_idx: &[usize],
    vtx2elem: &[usize],
    is_bidirectional: bool) -> (Vec<usize>, Vec<usize>) {
    let nvtx = vtx2elem_idx.len() - 1;
    const EDGES_PAR_TRI: [usize; 6] = [0, 1, 1, 2, 2, 0];
    const EDGES_PAR_QUAD: [usize; 8] = [0, 1, 1, 2, 2, 3, 3, 0];

    let mut vtx2vtx_idx = vec![0; nvtx + 1];
    let mut vtx2vtx = Vec::<usize>::new();

    for ip in 0..nvtx {
        let mut set_vtx_idx = std::collections::BTreeSet::new();
        for ielsup in vtx2elem_idx[ip]..vtx2elem_idx[ip + 1] {
            let ielem0 = vtx2elem[ielsup];
            let nnode = elem2vtx_index[ielem0 + 1] - elem2vtx_index[ielem0];
            let nedge = if nnode == 3 { 3 } else { 4 };
            for iedge in 0..nedge {
                let inode0: usize;
                let inode1: usize;
                {
                    if nnode == 3 {
                        inode0 = EDGES_PAR_TRI[iedge * 2 + 0];
                        inode1 = EDGES_PAR_TRI[iedge * 2 + 1];
                    } else {
                        inode0 = EDGES_PAR_QUAD[iedge * 2 + 0];
                        inode1 = EDGES_PAR_QUAD[iedge * 2 + 1];
                    }
                }
                let ip0 = elem2vtx[elem2vtx_index[ielem0] + inode0];
                let ip1 = elem2vtx[elem2vtx_index[ielem0] + inode1];
                if ip0 != ip && ip1 != ip { continue; }
                if ip0 == ip {
                    if is_bidirectional || ip1 > ip {
                        set_vtx_idx.insert(ip1);
                    }
                } else {
                    if is_bidirectional || ip0 > ip {
                        set_vtx_idx.insert(ip0);
                    }
                }
            }
        }
        for itr in &set_vtx_idx {
            vtx2vtx.push((*itr).try_into().unwrap());
        }
        vtx2vtx_idx[ip + 1] = vtx2vtx_idx[ip] + set_vtx_idx.len();
    }
    (vtx2vtx_idx, vtx2vtx)
}

pub fn meshtri_from_meshtriquad(
    elem2vtx_idx: &[usize],
    elem2vtx: &[usize]) -> Vec<usize> {
    let mut num_tri = 0_usize;
    for ielem in 0..elem2vtx_idx.len() - 1 {
        let nnode = elem2vtx_idx[ielem + 1] - elem2vtx_idx[ielem];
        if nnode == 3 { num_tri += 1; } else if nnode == 4 { num_tri += 2; }
    }
    let mut tri2vtx = Vec::<usize>::new();
    tri2vtx.reserve(num_tri * 3);
    for ielem in 0..elem2vtx_idx.len() - 1 {
        let nnode = elem2vtx_idx[ielem + 1] - elem2vtx_idx[ielem];
        let iiv0 = elem2vtx_idx[ielem];
        if nnode == 3 {
            tri2vtx.push(elem2vtx[iiv0 + 0]);
            tri2vtx.push(elem2vtx[iiv0 + 1]);
            tri2vtx.push(elem2vtx[iiv0 + 2]);
        } else if nnode == 4 {
            tri2vtx.push(elem2vtx[iiv0 + 0]);
            tri2vtx.push(elem2vtx[iiv0 + 1]);
            tri2vtx.push(elem2vtx[iiv0 + 2]);
            //
            tri2vtx.push(elem2vtx[iiv0 + 0]);
            tri2vtx.push(elem2vtx[iiv0 + 2]);
            tri2vtx.push(elem2vtx[iiv0 + 3]);
        }
    }
    tri2vtx
}

pub fn meshline_from_meshtriquad(
    elem2vtx_idx: &[usize],
    elem2vtx: &[usize],
    num_vtx: usize) -> Vec<usize> {
    use crate::topology_uniform::mshline_psup;
    let vtx2elem = elsup(
        &elem2vtx_idx, &elem2vtx,
        num_vtx);
    let vtx2vtx = psupedge_from_meshtriquad(
        &elem2vtx_idx, &elem2vtx,
        &vtx2elem.0, &vtx2elem.1,
        false);
     mshline_psup(&vtx2vtx.0, &vtx2vtx.1)
}