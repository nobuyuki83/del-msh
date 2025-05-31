//! unify the index of uv and xyz vertices for Wavefront Obj format

use num_traits::AsPrimitive;

/// * Returns
///   - (tri2uni, uni2vtxa, uni2vtxb)
pub fn unify_two_indices_of_triangle_mesh<Index>(
    tri2vtxa: &[Index],
    tri2vtxb: &[Index],
) -> (Vec<Index>, Vec<Index>, Vec<Index>)
where
    Index: num_traits::PrimInt + std::ops::AddAssign + 'static + AsPrimitive<usize>,
    usize: AsPrimitive<Index>,
{
    let num_vtxa = *tri2vtxa.iter().max().unwrap() + Index::one();
    let num_vtxb = *tri2vtxb.iter().max().unwrap() + Index::one();
    assert_eq!(tri2vtxa.len(), tri2vtxb.len());
    let vtxa2tri = crate::vtx2elem::from_uniform_mesh(tri2vtxa, 3, num_vtxa.as_());
    let vtxb2tri = crate::vtx2elem::from_uniform_mesh(tri2vtxb, 3, num_vtxb.as_());
    let num_tri = tri2vtxa.len() / 3;
    let mut tri2uni = vec![Index::max_value(); num_tri * 3];
    let mut uni2vtxa = Vec::<Index>::new();
    let mut uni2vtxb = Vec::<Index>::new();

    for i_tri in 0..num_tri {
        for i_node in 0..3 {
            if tri2uni[i_tri * 3 + i_node] != Index::max_value() {
                continue;
            }
            let i_vtxa = tri2vtxa[i_tri * 3 + i_node];
            let i_vtxb = tri2vtxb[i_tri * 3 + i_node];
            let s0 =
                &vtxa2tri.1[vtxa2tri.0[i_vtxa.as_()].as_()..vtxa2tri.0[i_vtxa.as_() + 1].as_()];
            let s1 =
                &vtxb2tri.1[vtxb2tri.0[i_vtxb.as_()].as_()..vtxb2tri.0[i_vtxb.as_() + 1].as_()];
            let s0 = std::collections::BTreeSet::<&Index>::from_iter(s0.iter());
            let s1 = std::collections::BTreeSet::<&Index>::from_iter(s1.iter());
            let intersection: Vec<_> = s0.intersection(&s1).collect();
            if intersection.is_empty() {
                continue;
            }
            let i_uni = uni2vtxb.len(); // new unified vertex
            assert_eq!(uni2vtxb.len(), uni2vtxa.len());
            uni2vtxa.push(i_vtxa);
            uni2vtxb.push(i_vtxb);
            for j_tri in intersection.into_iter().cloned() {
                let j_tri: usize = j_tri.as_();
                for j_node in 0..3 {
                    if tri2vtxa[j_tri * 3 + j_node] == i_vtxa
                        && tri2vtxb[j_tri * 3 + j_node] == i_vtxb
                    {
                        tri2uni[j_tri * 3 + j_node] = i_uni.as_();
                    }
                }
            }
        }
    }
    (tri2uni, uni2vtxa, uni2vtxb)
}

pub fn unify_two_indices_of_polygon_mesh(
    elem2idx: &[usize],
    idx2vtxa: &[usize],
    idx2vtxb: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let num_vtxa: usize = idx2vtxa.iter().max().unwrap() + 1;
    let num_vtxb: usize = idx2vtxb.iter().max().unwrap() + 1;
    let num_idx = idx2vtxa.len();
    assert_eq!(idx2vtxb.len(), num_idx);
    let vtxa2elem = crate::vtx2elem::from_polygon_mesh(elem2idx, idx2vtxa, num_vtxa);
    let vtxb2elem = crate::vtx2elem::from_polygon_mesh(elem2idx, idx2vtxb, num_vtxb);
    let num_elem = elem2idx.len() - 1;
    let mut idx2uni = vec![usize::MAX; num_idx];
    let mut uni2vtxa = Vec::<usize>::new();
    let mut uni2vtxb = Vec::<usize>::new();

    for i_elem in 0..num_elem {
        for idx in elem2idx[i_elem]..elem2idx[i_elem + 1] {
            if idx2uni[idx] != usize::MAX {
                continue;
            }
            let i_vtxa = idx2vtxa[idx];
            let i_vtxb = idx2vtxb[idx];
            let s0 = &vtxa2elem.1[vtxa2elem.0[i_vtxa]..vtxa2elem.0[i_vtxa + 1]];
            let s1 = &vtxb2elem.1[vtxb2elem.0[i_vtxb]..vtxb2elem.0[i_vtxb + 1]];
            let s0 = std::collections::BTreeSet::<&usize>::from_iter(s0.iter());
            let s1 = std::collections::BTreeSet::<&usize>::from_iter(s1.iter());
            let intersection: Vec<_> = s0.intersection(&s1).collect();
            assert!(!intersection.is_empty());
            let i_uni = uni2vtxb.len(); // new unified vertex
            assert_eq!(uni2vtxb.len(), uni2vtxa.len());
            uni2vtxa.push(i_vtxa);
            uni2vtxb.push(i_vtxb);
            assert!(intersection.clone().into_iter().any(|&&v| v == i_elem));
            // idx2uni[idx] = i_uni;
            for &j_elem in intersection.into_iter().cloned() {
                for jdx in elem2idx[j_elem]..elem2idx[j_elem + 1] {
                    if idx2vtxa[jdx] == i_vtxa && idx2vtxb[jdx] == i_vtxb {
                        idx2uni[jdx] = i_uni;
                    }
                }
            }
        }
    }
    (idx2uni, uni2vtxa, uni2vtxb)
}
