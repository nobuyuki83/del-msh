/// element surrounding point (elsup)
pub fn elsup(
    elem_vtx_idx: &[usize],
    elem_vtx: &[usize],
    nvtx: usize) -> (Vec<usize>, Vec<usize>)
{
    let nelem = elem_vtx_idx.len() - 1;
    let mut elsup_ind = Vec::new();
    elsup_ind.resize(nvtx + 1, 0);
    for ielem in 0..nelem {
        for iivtx in elem_vtx_idx[ielem]..elem_vtx_idx[ielem + 1] {
            let ivtx0 = elem_vtx[iivtx];
            let ivtx0: usize = ivtx0 as usize;
            elsup_ind[ivtx0 + 1] += 1;
        }
    }
    for ivtx in 0..nvtx {
        elsup_ind[ivtx + 1] += elsup_ind[ivtx];
    }
    let nelsup = elsup_ind[nvtx];
    let mut elsup = Vec::new();
    elsup.resize(nelsup, 0);
    for ielem in 0..nelem {
        for iivtx in elem_vtx_idx[ielem]..elem_vtx_idx[ielem + 1] {
            let ivtx0 = elem_vtx[iivtx];
            let ivtx0 = ivtx0 as usize;
            let ind1 = elsup_ind[ivtx0];
            elsup[ind1] = ielem;
            elsup_ind[ivtx0] += 1;
        }
    }
    for ivtx in (1..nvtx).rev() {
        elsup_ind[ivtx] = elsup_ind[ivtx - 1];
    }
    elsup_ind[0] = 0;
    (elsup_ind, elsup)
}

#[test]
fn test_psup_mshmix() {
    let elem_vtx_idx = vec![0, 3, 7, 11, 14];
    let elem_vtx = vec![0, 4, 2, 4, 3, 5, 2, 1, 6, 7, 5, 3, 1, 5];
    let (elsup_ind, elsup) = elsup(&elem_vtx_idx, &elem_vtx, 8);
    assert_eq!(elsup_ind, vec![0, 1, 3, 5, 7, 9, 12, 13, 14]);
    assert_eq!(elsup, vec![0, 2, 3, 0, 1, 1, 3, 0, 1, 1, 2, 3, 2, 2]);
}

pub fn psupedge_from_meshtriquad(
    elem_vtx_index: &[usize],
    elem_vtx: &[usize],
    elsup_ind: &Vec<usize>,
    elsup: &Vec<usize>,
    is_bidirectional: bool) -> (Vec<usize>, Vec<usize>) {
    let nvtx = elsup_ind.len() - 1;
    const EDGES_PAR_TRI: [usize; 6] = [0, 1, 1, 2, 2, 0];
    const EDGES_PAR_QUAD: [usize; 8] = [0, 1, 1, 2, 2, 3, 3, 0];

    let mut psup = Vec::<usize>::new();
    let mut psup_ind = vec![0; nvtx + 1];

    for ip in 0..nvtx {
        let mut set_vtx_idx = std::collections::BTreeSet::new();
        for ielsup in elsup_ind[ip]..elsup_ind[ip + 1] {
            let ielem0 = elsup[ielsup];
            let nnode = elem_vtx_index[ielem0 + 1] - elem_vtx_index[ielem0];
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
                let ip0 = elem_vtx[elem_vtx_index[ielem0] + inode0];
                let ip1 = elem_vtx[elem_vtx_index[ielem0] + inode1];
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
            psup.push((*itr).try_into().unwrap());
        }
        psup_ind[ip + 1] = psup_ind[ip] + set_vtx_idx.len();
    }
    (psup_ind, psup)
}

pub fn meshtri_from_meshtriquad(
    elem_vtx_index: &[usize],
    elem_vtx: &[usize]) -> Vec<usize> {
    let mut ntri = 0_usize;
    for ielem in 0..elem_vtx_index.len() - 1 {
        let nnode = elem_vtx_index[ielem + 1] - elem_vtx_index[ielem];
        if nnode == 3 { ntri += 1; } else if nnode == 4 { ntri += 2; }
    }
    let mut tri_vtx = Vec::<usize>::new();
    tri_vtx.reserve(ntri * 3);
    for ielem in 0..elem_vtx_index.len() - 1 {
        let nnode = elem_vtx_index[ielem + 1] - elem_vtx_index[ielem];
        let iiv0 = elem_vtx_index[ielem];
        if nnode == 3 {
            tri_vtx.push(elem_vtx[iiv0 + 0]);
            tri_vtx.push(elem_vtx[iiv0 + 1]);
            tri_vtx.push(elem_vtx[iiv0 + 2]);
        } else if nnode == 4 {
            tri_vtx.push(elem_vtx[iiv0 + 0]);
            tri_vtx.push(elem_vtx[iiv0 + 1]);
            tri_vtx.push(elem_vtx[iiv0 + 2]);
            //
            tri_vtx.push(elem_vtx[iiv0 + 0]);
            tri_vtx.push(elem_vtx[iiv0 + 2]);
            tri_vtx.push(elem_vtx[iiv0 + 3]);
        }
    }
    tri_vtx
}

pub fn meshline_from_meshtriquad(
    elem_vtx_index: &[usize],
    elem_vtx: &[usize],
    num_vtx: usize)  -> Vec<usize> {
    use crate::topology_uniform::mshline_psup;
    let elsup = elsup(
        &elem_vtx_index, &elem_vtx,
        num_vtx);
    let psup = psupedge_from_meshtriquad(
        &elem_vtx_index, &elem_vtx,
        &elsup.0, &elsup.1,
        false);
     mshline_psup(&psup.0, &psup.1)
}