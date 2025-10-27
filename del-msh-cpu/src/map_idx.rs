//! methods for mapping data on vertex

use num_traits::AsPrimitive;

pub fn from_remove_unreferenced_vertices(
    elem2vtxa: &[usize],
    num_vtxa: usize,
) -> (Vec<usize>, usize) {
    let mut vtxa2vtxb = vec![usize::MAX; num_vtxa];
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

pub fn map_vertex_attibute_to<T>(
    vtxa2xyz: &[T],
    num_dim: usize,
    vtxa2vtxb: &[usize],
    num_vtxb: usize,
) -> Vec<T>
where
    T: Copy + num_traits::Zero,
{
    let num_vtxa = vtxa2xyz.len() / num_dim;
    let mut vtxb2xyz = vec![T::zero(); num_vtxb * num_dim];
    for i_vtxa in 0..num_vtxa {
        let i_vtxb = vtxa2vtxb[i_vtxa];
        if i_vtxb == usize::MAX {
            continue;
        }
        for i_dim in 0..num_dim {
            vtxb2xyz[i_vtxb * num_dim + i_dim] = vtxa2xyz[i_vtxa * num_dim + i_dim];
        }
    }
    vtxb2xyz
}

pub fn map_vertex_attibute_from<Real, Index>(
    vtxa2xyz: &[Real],
    num_dim: usize,
    vtxb2vtxa: &[Index],
) -> Vec<Real>
where
    Real: Copy + num_traits::Zero,
    Index: num_traits::PrimInt + AsPrimitive<usize>,
{
    let num_vtxb = vtxb2vtxa.len();
    let mut vtxb2xyz = vec![Real::zero(); num_vtxb * num_dim];
    for i_vtxb in 0..num_vtxb {
        let i_vtxa = vtxb2vtxa[i_vtxb];
        if i_vtxa == Index::max_value() {
            continue;
        }
        let i_vtxa: usize = i_vtxa.as_();
        for i_dim in 0..num_dim {
            vtxb2xyz[i_vtxb * num_dim + i_dim] = vtxa2xyz[i_vtxa * num_dim + i_dim];
        }
    }
    vtxb2xyz
}

pub fn map_elem_index(elem2vtxa: &[usize], vtxa2vtxb: &[usize]) -> Vec<usize> {
    let mut elem2vtxb = vec![0usize; elem2vtxa.len()];
    for it in 0..elem2vtxa.len() {
        let i_vtxa = elem2vtxa[it];
        let i_vtxb = vtxa2vtxb[i_vtxa];
        elem2vtxb[it] = i_vtxb;
    }
    elem2vtxb
}

/// "old2new" should be sorted
/// this function use bisection algorithm to find array of offset
pub fn inverse(old2new: &[u32], new2old_offset: &mut [u32]) {
    let num_new = new2old_offset.len() - 1;
    let num_old = old2new.len();
    for i_new in 0..num_new + 1 {
        let mut i0_old = 0;
        let mut i2_old = num_old;
        new2old_offset[i_new] = loop {
            if i2_old - i0_old == 1 {
                if old2new[i0_old] as usize >= i_new {
                    break i0_old as u32;
                } else {
                    break i2_old as u32;
                }
            }
            let i1_old = (i2_old + i0_old) / 2;
            if (old2new[i1_old] as usize) < i_new {
                i0_old = i1_old;
            } else {
                i2_old = i1_old;
            }
        };
    }
}
