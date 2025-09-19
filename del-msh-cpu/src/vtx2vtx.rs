//! methods that generate vertices connected to a vertex

use num_traits::{AsPrimitive, PrimInt};

/// point surrounding point for mesh
/// * `elem2vtx` - map element to vertex: list of vertex index for each element
/// * `num_node` - number of vertex par element (e.g., 3 for tri, 4 for quad)
/// * `num_vtx` - number of vertex
/// * `vtx2idx` - map vertex to element index: cumulative sum
/// * `idx2elem` - map vertex to element value: list of value
pub fn from_uniform_mesh_with_vtx2elem<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    num_vtx: usize,
    vtx2idx: &[Index],
    idx2elem: &[Index],
    is_self: bool,
) -> (Vec<Index>, Vec<Index>)
where
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize> + std::ops::AddAssign<Index>,
    usize: AsPrimitive<Index>,
{
    assert_eq!(vtx2idx.len(), num_vtx + 1);
    assert_eq!(elem2vtx.len() % num_node, 0);
    let mut vtx2flg = vec![usize::MAX; num_vtx];
    let mut vtx2jdx = vec![Index::zero(); num_vtx + 1];
    for i_vtx in 0..num_vtx {
        if !is_self {
            vtx2flg[i_vtx] = i_vtx;
        }
        let idx0: usize = vtx2idx[i_vtx].as_();
        let idx1: usize = vtx2idx[i_vtx + 1].as_();
        for j_elem in &idx2elem[idx0..idx1] {
            let j_elem: usize = j_elem.as_();
            for j_node in 0..num_node {
                let j_vtx: usize = elem2vtx[j_elem * num_node + j_node].as_();
                if vtx2flg[j_vtx] != i_vtx {
                    vtx2flg[j_vtx] = i_vtx;
                    vtx2jdx[i_vtx + 1] += Index::one();
                }
            }
        }
    }
    for i_vtx in 0..num_vtx {
        let tmp = vtx2jdx[i_vtx];
        vtx2jdx[i_vtx + 1] += tmp;
    }
    let num_vtx2vtx: usize = vtx2jdx[num_vtx].as_();
    let mut jdx2vtx = vec![Index::zero(); num_vtx2vtx];
    vtx2flg.iter_mut().for_each(|v| *v = usize::MAX);
    for i_vtx in 0..num_vtx {
        if !is_self {
            vtx2flg[i_vtx] = i_vtx;
        }
        let idx0: usize = vtx2idx[i_vtx].as_();
        let idx1: usize = vtx2idx[i_vtx + 1].as_();
        for j_elem in &idx2elem[idx0..idx1] {
            let j_elem: usize = j_elem.as_();
            for j_node in 0..num_node {
                let j_vtx: usize = elem2vtx[j_elem * num_node + j_node].as_();
                if vtx2flg[j_vtx] != i_vtx {
                    vtx2flg[j_vtx] = i_vtx;
                    let iv2v: usize = vtx2jdx[i_vtx].as_();
                    jdx2vtx[iv2v] = j_vtx.as_();
                    vtx2jdx[i_vtx] += Index::one();
                }
            }
        }
    }
    for i_vtx in (1..num_vtx).rev() {
        vtx2jdx[i_vtx] = vtx2jdx[i_vtx - 1];
    }
    vtx2jdx[0] = Index::zero();
    (vtx2jdx, jdx2vtx)
}

/// compute index of vertices adjacent to vertices for uniform mesh.
pub fn from_uniform_mesh<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    num_vtx: usize,
    is_self: bool,
) -> (Vec<Index>, Vec<Index>)
where
    Index: num_traits::PrimInt + std::ops::AddAssign + num_traits::AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    // set pattern to sparse matrix
    assert_eq!(elem2vtx.len() % num_node, 0);
    let vtx2elem = crate::vtx2elem::from_uniform_mesh(elem2vtx, num_node, num_vtx);
    assert_eq!(vtx2elem.0.len(), num_vtx + 1);
    let vtx2vtx = from_uniform_mesh_with_vtx2elem(
        elem2vtx,
        num_node,
        num_vtx,
        &vtx2elem.0,
        &vtx2elem.1,
        is_self,
    );
    assert_eq!(vtx2vtx.0.len(), num_vtx + 1);
    vtx2vtx
}

pub fn from_specific_edges_of_uniform_mesh<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    edge2node: &[usize],
    vtx2idx: &[Index],
    idx2elem: &[Index],
    is_bidirectional: bool,
) -> (Vec<Index>, Vec<Index>)
where
    Index: num_traits::PrimInt + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let num_edge = edge2node.len() / 2;
    assert_eq!(edge2node.len(), num_edge * 2);

    let num_vtx = vtx2idx.len() - 1;
    let mut vtx2jdx = vec![Index::zero(); num_vtx + 1];
    vtx2jdx[0] = Index::zero();
    let mut jdx2vtx = Vec::<Index>::new();
    let mut set_vtx: std::collections::BTreeSet<Index> = std::collections::BTreeSet::new();
    for i_vtx in 0..num_vtx {
        set_vtx.clear();
        let idx0: usize = vtx2idx[i_vtx].as_();
        let idx1: usize = vtx2idx[i_vtx + 1].as_();
        for &ielem0 in &idx2elem[idx0..idx1] {
            let i_vtx: Index = i_vtx.as_();
            let ielem0 = ielem0.as_();
            for iedge in 0..num_edge {
                let inode0 = edge2node[iedge * 2];
                let inode1 = edge2node[iedge * 2 + 1];
                let ivtx0 = elem2vtx[ielem0 * num_node + inode0];
                let ivtx1 = elem2vtx[ielem0 * num_node + inode1];
                if ivtx0 != i_vtx && ivtx1 != i_vtx {
                    continue;
                }
                if ivtx0 == i_vtx {
                    if is_bidirectional || ivtx1 > i_vtx {
                        set_vtx.insert(ivtx1);
                    }
                } else if is_bidirectional || ivtx0 > i_vtx {
                    set_vtx.insert(ivtx0);
                }
            }
        }
        for vtx in &set_vtx {
            jdx2vtx.push(*vtx);
        }
        vtx2jdx[i_vtx + 1] = vtx2jdx[i_vtx] + set_vtx.len().as_();
    }
    (vtx2jdx, jdx2vtx)
}

/// make vertex surrounding vertex as edges of polygon mesh.
/// A polygon mesh is a mixture of elements such as triangle, quadrilateal, pentagon.
pub fn from_polygon_mesh_edges_with_vtx2elem(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    vtx2jdx: &[usize],
    jdx2elem: &[usize],
    is_bidirectional: bool,
) -> (Vec<usize>, Vec<usize>) {
    let nvtx = vtx2jdx.len() - 1;

    let mut vtx2kdx = vec![0; nvtx + 1];
    let mut kdx2vtx = Vec::<usize>::new();

    for i_vtx in 0..nvtx {
        let mut set_vtx_idx = std::collections::BTreeSet::new();
        for &ielem0 in &jdx2elem[vtx2jdx[i_vtx]..vtx2jdx[i_vtx + 1]] {
            let num_node = elem2idx[ielem0 + 1] - elem2idx[ielem0];
            let num_edge = num_node;
            for i_edge in 0..num_edge {
                let i_node0 = i_edge;
                let i_node1 = (i_edge + 1) % num_node;
                let j_vtx0 = idx2vtx[elem2idx[ielem0] + i_node0];
                let j_vtx1 = idx2vtx[elem2idx[ielem0] + i_node1];
                if j_vtx0 != i_vtx && j_vtx1 != i_vtx {
                    continue;
                }
                if j_vtx0 == i_vtx {
                    if is_bidirectional || j_vtx1 > i_vtx {
                        set_vtx_idx.insert(j_vtx1);
                    }
                } else if is_bidirectional || j_vtx0 > i_vtx {
                    set_vtx_idx.insert(j_vtx0);
                }
            }
        }
        for itr in &set_vtx_idx {
            kdx2vtx.push(*itr);
        }
        vtx2kdx[i_vtx + 1] = vtx2kdx[i_vtx] + set_vtx_idx.len();
    }
    (vtx2kdx, kdx2vtx)
}

/// \[I + lambda * L\] {vtx2lhs} = {vtx2rhs}
/// L = \[..-1,..,valence, ..,-1 \]
pub fn laplacian_smoothing<const NDIM: usize, IDX>(
    vtx2idx: &[IDX],
    idx2vtx: &[IDX],
    lambda: f32,
    vtx2lhs: &mut [f32],
    vtx2rhs: &[f32],
    num_iter: usize,
    vtx2lhs_tmp: &mut [f32],
) where
    IDX: num_traits::PrimInt + AsPrimitive<usize> + AsPrimitive<f32> + std::marker::Sync,
{
    let num_vtx = vtx2idx.len() - 1;
    assert_eq!(vtx2lhs.len(), num_vtx * NDIM);
    assert_eq!(vtx2rhs.len(), num_vtx * NDIM);
    assert_eq!(vtx2lhs_tmp.len(), num_vtx * NDIM);
    let func_upd = |i_vtx: usize, lhs_next: &mut [f32], vtx2lhs_prev: &[f32]| {
        let mut rhs: [f32; NDIM] = std::array::from_fn(|i| vtx2rhs[i_vtx * NDIM + i]);
        for &j_vtx in &idx2vtx[vtx2idx[i_vtx].as_()..vtx2idx[i_vtx + 1].as_()] {
            let j_vtx: usize = j_vtx.as_();
            for i in 0..NDIM {
                rhs[i] += lambda * vtx2lhs_prev[j_vtx * NDIM + i];
            }
        }
        let valence: f32 = (vtx2idx[i_vtx + 1] - vtx2idx[i_vtx]).as_();
        let inv_dia = 1f32 / (1f32 + lambda * valence);
        for i in 0..NDIM {
            lhs_next[i] = rhs[i] * inv_dia;
        }
    };
    use rayon::prelude::*;
    for _iter in 0..num_iter {
        vtx2lhs_tmp
            .par_chunks_mut(NDIM)
            .enumerate()
            .for_each(|(i_vtx, lhs1)| func_upd(i_vtx, lhs1, &vtx2lhs));
        vtx2lhs
            .par_chunks_mut(NDIM)
            .enumerate()
            .for_each(|(i_vtx, lhs)| func_upd(i_vtx, lhs, &vtx2lhs_tmp));
    }
}

pub fn compute_residual_norm_of_laplacian_smoothing<const NDIM: usize, IDX>(
    vtx2idx: &[IDX],
    idx2vtx: &[IDX],
    vtx2rhs: &[f32],
    vtx2lhs: &[f32],
    lambda: f32,
) -> f32
where
    IDX: PrimInt + AsPrimitive<usize> + AsPrimitive<f32>,
{
    let num_vtx = vtx2rhs.len() / 3;
    let func_res = |i_vtx: usize| -> f32 {
        let mut res: [f32; NDIM] = std::array::from_fn(|i| vtx2rhs[i_vtx * NDIM + i]);
        for &j_vtx in &idx2vtx[vtx2idx[i_vtx].as_()..vtx2idx[i_vtx + 1].as_()] {
            let j_vtx: usize = j_vtx.as_();
            for i in 0..NDIM {
                res[i] += lambda * vtx2lhs[j_vtx * NDIM + i];
            }
        }
        let valence: f32 = (vtx2idx[i_vtx + 1] - vtx2idx[i_vtx]).as_();
        for i in 0..NDIM {
            res[i] -= (1f32 + lambda * valence) * vtx2lhs[i_vtx * NDIM + i];
        }
        res.iter().map(|v| v * v).sum::<f32>()
    };
    (0..num_vtx).map(func_res).sum()
}

#[test]
fn test_laplacian_smoothing() {
    let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::torus_zup::<usize, f32>(1.0, 0.3, 32, 32);
    let (vtx2idx, idx2vtx) =
        crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, vtx2xyz.len() / 3, false);
    let vtx2rhs = {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        (0..vtx2xyz.len())
            .map(|_| rng.random())
            .collect::<Vec<f32>>()
    };
    let lambda = 1f32;
    let mut vtx2lhs = vec![0f32; vtx2xyz.len()];
    let res0 = compute_residual_norm_of_laplacian_smoothing::<3, usize>(
        &vtx2idx, &idx2vtx, &vtx2rhs, &vtx2lhs, lambda,
    );
    assert!(res0 > 1000.);
    {
        let mut vtx2lhs_tmp = vtx2lhs.clone();
        laplacian_smoothing::<3, usize>(
            &vtx2idx,
            &idx2vtx,
            lambda,
            &mut vtx2lhs,
            &vtx2rhs,
            100,
            &mut vtx2lhs_tmp,
        );
    }
    let res1 = compute_residual_norm_of_laplacian_smoothing::<3, usize>(
        &vtx2idx, &idx2vtx, &vtx2rhs, &vtx2lhs, lambda,
    );
    assert!(res1 < 1.0e-9);
    dbg!(res0, res1);
}
