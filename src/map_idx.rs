//! methods for all kinds of mesh

pub fn from_remove_unreferenced_vertices(
    elem2vtxa: &[usize],
    num_vtxa: usize) -> (Vec<usize>, usize)
{
    let mut vtxa2vtxb = vec!(usize::MAX; num_vtxa);
    for &i_vtxa in elem2vtxa {
        vtxa2vtxb[i_vtxa] = 0;
    }
    let mut i_vtxb = 0;
    for j_vtxb in &mut vtxa2vtxb {
        if *j_vtxb == usize::MAX {
            continue;
        }
        *j_vtxb = i_vtxb;
        i_vtxb += 1;
    }
    (vtxa2vtxb, i_vtxb)
}


pub fn map_vertex_attibute<T>(
    vtxa2xyz: &[T],
    num_dim: usize,
    vtxa2vtxb: &[usize],
    num_vtxb: usize) -> Vec<T>
    where T: Copy + num_traits::Zero
{
    let num_vtxa = vtxa2xyz.len() / num_dim;
    let mut vtxb2xyz = vec!(T::zero(); num_vtxb * num_dim);
    for i_vtxa in 0..num_vtxa {
        if vtxa2vtxb[i_vtxa] == usize::MAX {
            continue;
        }
        let jp = vtxa2vtxb[i_vtxa];
        for i_dim in 0..num_dim {
            vtxb2xyz[jp * num_dim + i_dim] = vtxa2xyz[i_vtxa * num_dim + i_dim];
        }
    }
    vtxb2xyz
}

pub fn map_elem_index(
    elem2vtxa: &[usize],
    vtxa2vtxb: &[usize]) -> Vec<usize>
{
    let mut elem2vtxb = vec!(0usize; elem2vtxa.len());
    for it in 0..elem2vtxa.len() {
        let i_vtxa = elem2vtxa[it];
        let i_vtxb = vtxa2vtxb[i_vtxa];
        elem2vtxb[it] = i_vtxb;
    }
    elem2vtxb
}