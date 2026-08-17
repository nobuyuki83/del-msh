//! extract subset of elements as mesh

pub fn extract(
    oldtri2oldvtx: &[[usize; 3]],
    num_oldvtx: usize,
    oldtri2newtri: &[usize],
    num_newtri: usize,
) -> (Vec<[usize; 3]>, usize, Vec<usize>) {
    assert_eq!(oldtri2oldvtx.len(), oldtri2newtri.len());
    let num_tri = oldtri2oldvtx.len();
    let mut oldvtx2newvtx = vec![usize::MAX; num_oldvtx];
    let mut newtri2newvtx = vec![[usize::MAX; 3]; num_newtri];
    let mut num_newvtx = 0;
    for i_tri in 0..num_tri {
        if oldtri2newtri[i_tri] == usize::MAX {
            continue;
        }
        let i_tri_new = oldtri2newtri[i_tri];
        for i_node in 0..3 {
            let i_vtx_xyz = oldtri2oldvtx[i_tri][i_node];
            if oldvtx2newvtx[i_vtx_xyz] == usize::MAX {
                let i_vtx_new = num_newvtx;
                num_newvtx += 1;
                oldvtx2newvtx[i_vtx_xyz] = i_vtx_new;
                newtri2newvtx[i_tri_new][i_node] = i_vtx_new;
            } else {
                let i_vtx_new = oldvtx2newvtx[i_vtx_xyz];
                newtri2newvtx[i_tri_new][i_node] = i_vtx_new;
            }
        }
    }
    (newtri2newvtx, num_newvtx, oldvtx2newvtx)
}

pub fn map_values_old2new<VALUE>(
    old2value: &[VALUE],
    od2new: &[usize],
    num_new: usize,
    num_dim: usize,
) -> Vec<VALUE>
where
    VALUE: num_traits::Zero + Copy,
{
    let mut new2value = vec![VALUE::zero(); num_new * num_dim];
    for i_old in 0..old2value.len() / num_dim {
        let i_new = od2new[i_old];
        if i_new == usize::MAX {
            continue;
        }
        for i_dim in 0..num_dim {
            new2value[i_new * num_dim + i_dim] = old2value[i_old * num_dim + i_dim];
        }
    }
    new2value
}

pub fn from_polygonal_mesh_array(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    elem2flag: &[bool],
) -> (Vec<usize>, Vec<usize>) {
    assert_eq!(elem2idx.len(), elem2flag.len() + 1);
    let mut felem2jdx = vec![0_usize; 1];
    let mut jdx2vtx = vec![0_usize; 0];
    for i_elem in 0..elem2flag.len() {
        if !elem2flag[i_elem] {
            continue;
        }
        let idx0 = elem2idx[i_elem];
        let idx1 = elem2idx[i_elem + 1];
        for &vtx in &idx2vtx[idx0..idx1] {
            jdx2vtx.push(vtx);
        }
        felem2jdx.push(jdx2vtx.len());
    }
    (felem2jdx, jdx2vtx)
}

pub fn from_polygonal_mesh_lambda<F: Fn(usize) -> bool>(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    elem2flag: F,
) -> (Vec<usize>, Vec<usize>) {
    let mut felem2jdx = vec![0_usize; 1];
    let mut jdx2vtx = vec![0_usize; 0];
    let num_elem = elem2idx.len() - 1;
    for i_elem in 0..num_elem {
        if !elem2flag(i_elem) {
            continue;
        }
        let idx0 = elem2idx[i_elem];
        let idx1 = elem2idx[i_elem + 1];
        for &vtx in &idx2vtx[idx0..idx1] {
            jdx2vtx.push(vtx);
        }
        felem2jdx.push(jdx2vtx.len());
    }
    (felem2jdx, jdx2vtx)
}

pub fn from_uniform_mesh_lambda<F: Fn(usize) -> bool>(
    elem2vtx: &[usize],
    num_node: usize,
    elem2flag: F,
) -> Vec<usize> {
    let num_elem = elem2vtx.len() / num_node;
    let mut felem2vtx = vec![0_usize; 0];
    for i_elem in 0..num_elem {
        if !elem2flag(i_elem) {
            continue;
        }
        for i_node in 0..num_node {
            felem2vtx.push(elem2vtx[i_elem * num_node + i_node]);
        }
    }
    felem2vtx
}

pub fn from_uniform_mesh_from_list_of_elements(
    elem2vtx: &[usize],
    num_node: usize,
    elems: &[usize],
) -> Vec<usize> {
    let mut felem2vtx = Vec::<usize>::with_capacity(elems.len() * num_node);
    for i_elem in elems {
        for i_node in 0..num_node {
            felem2vtx.push(elem2vtx[i_elem * num_node + i_node]);
        }
    }
    felem2vtx
}
