/// Compute the volume of each element in a mixed polyhedron mesh.
///
/// Supported element types (determined by node count per element):
/// - 4 nodes: tetrahedron
/// - 5 nodes: pyramid (square base + apex)
/// - 6 nodes: triangular prism
///
/// # Arguments
/// * `elem2idx_offset` - CSR offset array of length `num_elem + 1`; element `i` owns
///   vertices `idx2vtx[elem2idx_offset[i]..elem2idx_offset[i+1]]`
/// * `idx2vtx` - flat list of vertex indices for all elements
/// * `vtx2xyz` - vertex positions, stored as `[x, y, z, x, y, z, ...]`
/// * `i_gauss_degree` - Gauss quadrature degree used for pyramid and prism volume integration
/// * `elem2volume` - output slice of length `num_elem`; each entry is set to the element's volume
pub fn elem2volume<IDX, Real>(
    elem2idx_offset: &[IDX],
    idx2vtx: &[IDX],
    vtx2xyz: &[Real],
    i_gauss_degree: usize,
    elem2volume: &mut [Real],
) where
    IDX: num_traits::AsPrimitive<usize>,
    Real: num_traits::Float + 'static + std::fmt::Debug,
    del_geo_core::quadrature_line::Quad<Real>: del_geo_core::quadrature_line::QuadratureLine<Real>,
    del_geo_core::quadrature_tri::Quad<Real>: del_geo_core::quadrature_tri::QuadratureTri<Real>,
{
    let num_elem = elem2idx_offset.len() - 1;
    assert_eq!(elem2volume.len(), num_elem);
    for i_elem in 0..elem2idx_offset.len() - 1 {
        let node2vtx = &idx2vtx[elem2idx_offset[i_elem].as_()..elem2idx_offset[i_elem + 1].as_()];
        let p = |i: usize| arrayref::array_ref![vtx2xyz, node2vtx[i].as_() * 3, 3];
        match node2vtx.len() {
            4 => {
                elem2volume[i_elem] = del_geo_core::tet::volume(p(0), p(1), p(2), p(3));
            }
            5 => {
                elem2volume[i_elem] =
                    del_geo_core::pyramid::volume(p(0), p(1), p(2), p(3), p(4), i_gauss_degree);
            }
            6 => {
                elem2volume[i_elem] =
                    del_geo_core::prism::volume(p(0), p(1), p(2), p(3), p(4), p(5), i_gauss_degree);
            }
            _ => {
                unreachable!()
            }
        }
    }
}

/// Returns the nearest point on the element and the 3 parametric coordinates at that point.
/// - Tet (4):    (r0, r1, r2); r3 = 1-r0-r1-r2 is implicit
/// - Pyramid (5): (r, s, t) with r,s in [0,1], t in [0,1]
/// - Prism (6):   (r, s, t) with r,s>=0, r+s<=1, t in [0,1]
fn nearest_to_elem(query: &[f32; 3], node2vtx: &[u32], vtx2xyz: &[f32]) -> ([f32; 3], [f32; 3]) {
    let shift = |i: usize| -> [f32; 3] {
        let p = arrayref::array_ref!(vtx2xyz, i * 3, 3);
        [p[0] - query[0], p[1] - query[1], p[2] - query[2]]
    };
    match node2vtx.len() {
        4 => {
            let (v0, v1, v2, v3) = (
                shift(node2vtx[0] as usize),
                shift(node2vtx[1] as usize),
                shift(node2vtx[2] as usize),
                shift(node2vtx[3] as usize),
            );
            let (p, r0, r1, r2) = del_geo_core::tet::nearest_to_origin(&v0, &v1, &v2, &v3);
            let np = [p[0] + query[0], p[1] + query[1], p[2] + query[2]];
            (np, [r0, r1, r2])
        }
        5 => {
            let (v0, v1, v2, v3, v4) = (
                shift(node2vtx[0] as usize),
                shift(node2vtx[1] as usize),
                shift(node2vtx[2] as usize),
                shift(node2vtx[3] as usize),
                shift(node2vtx[4] as usize),
            );
            // shapefunc([r,s,t]) = [(1-r)(1-s)(1-t), r(1-s)(1-t), rs(1-t), (1-r)s(1-t), t]
            // inverse: t=w4, r=(w1+w2)/(1-t), s=(w2+w3)/(1-t)
            let (p, w) = del_geo_core::pyramid::nearest_to_origin(&v0, &v1, &v2, &v3, &v4);
            let t = w[4];
            let one_mt = 1.0 - t;
            let (r, s) = if one_mt > 1e-6 {
                ((w[1] + w[2]) / one_mt, (w[2] + w[3]) / one_mt)
            } else {
                (0.5, 0.5) // at apex, r/s undefined; use centre of base
            };
            let np = [p[0] + query[0], p[1] + query[1], p[2] + query[2]];
            (np, [r, s, t])
        }
        6 => {
            let (v0, v1, v2, v3, v4, v5) = (
                shift(node2vtx[0] as usize),
                shift(node2vtx[1] as usize),
                shift(node2vtx[2] as usize),
                shift(node2vtx[3] as usize),
                shift(node2vtx[4] as usize),
                shift(node2vtx[5] as usize),
            );
            // shapefunc([r,s,t]) = [(1-r-s)(1-t), r(1-t), s(1-t), (1-r-s)t, rt, st]
            // inverse: t=w3+w4+w5, r=w1/(1-t) or w4/t, s=w2/(1-t) or w5/t
            let (p, w) = del_geo_core::prism::nearest_to_origin(&v0, &v1, &v2, &v3, &v4, &v5);
            let t = w[3] + w[4] + w[5];
            let one_mt = 1.0 - t;
            let (r, s) = if one_mt > 1e-6 {
                (w[1] / one_mt, w[2] / one_mt)
            } else if t > 1e-6 {
                (w[4] / t, w[5] / t)
            } else {
                (1.0 / 3.0, 1.0 / 3.0)
            };
            let np = [p[0] + query[0], p[1] + query[1], p[2] + query[2]];
            (np, [r, s, t])
        }
        n => panic!("unsupported element type with {n} vertices"),
    }
}

fn nearest_elem_bvh(
    query: &[f32; 3],
    bvhnodes: &[u32],
    bvhnode2aabb: &[f32],
    elem2idx_offset: &[u32],
    idx2vtx: &[u32],
    vtx2xyz: &[f32],
    i_bvhnode: usize,
    best_dist_sq: &mut f32,
    best_elem: &mut usize,
    best_weights: &mut [f32; 3],
) {
    let aabb = arrayref::array_ref!(bvhnode2aabb, i_bvhnode * 6, 6);
    if del_geo_core::aabb3::min_sq_dist_to_point3(aabb, query) >= *best_dist_sq {
        return;
    }
    if bvhnodes[i_bvhnode * 3 + 2] == u32::MAX {
        // leaf node
        let i_elem = bvhnodes[i_bvhnode * 3 + 1] as usize;
        let i0 = elem2idx_offset[i_elem] as usize;
        let i1 = elem2idx_offset[i_elem + 1] as usize;
        let (np, weights) = nearest_to_elem(query, &idx2vtx[i0..i1], vtx2xyz);
        let dx = np[0] - query[0];
        let dy = np[1] - query[1];
        let dz = np[2] - query[2];
        let d = dx * dx + dy * dy + dz * dz;
        if d < *best_dist_sq {
            *best_dist_sq = d;
            *best_elem = i_elem;
            *best_weights = weights;
        }
    } else {
        let i_child0 = bvhnodes[i_bvhnode * 3 + 1] as usize;
        let i_child1 = bvhnodes[i_bvhnode * 3 + 2] as usize;
        let d0 = del_geo_core::aabb3::min_sq_dist_to_point3(
            arrayref::array_ref!(bvhnode2aabb, i_child0 * 6, 6),
            query,
        );
        let d1 = del_geo_core::aabb3::min_sq_dist_to_point3(
            arrayref::array_ref!(bvhnode2aabb, i_child1 * 6, 6),
            query,
        );
        let (first, second) = if d0 <= d1 {
            (i_child0, i_child1)
        } else {
            (i_child1, i_child0)
        };
        nearest_elem_bvh(
            query,
            bvhnodes,
            bvhnode2aabb,
            elem2idx_offset,
            idx2vtx,
            vtx2xyz,
            first,
            best_dist_sq,
            best_elem,
            best_weights,
        );
        nearest_elem_bvh(
            query,
            bvhnodes,
            bvhnode2aabb,
            elem2idx_offset,
            idx2vtx,
            vtx2xyz,
            second,
            best_dist_sq,
            best_elem,
            best_weights,
        );
    }
}

/// For each query point in `wtx2xyz` (shape N×3, flat), find the nearest polyhedron element
/// using BVH acceleration. Returns `(wtx2elem, wtx2param)` where:
/// - `wtx2elem`: (N,) - nearest element index per query point
/// - `wtx2param`: (N×6, flat) - shape function weights within the nearest element;
///   padded with zeros for element types with fewer than 6 nodes
pub fn nearest_elem_for_points(
    bvhnodes: &[u32],
    bvhnode2aabb: &[f32],
    elem2idx_offset: &[u32],
    idx2vtx: &[u32],
    vtx2xyz: &[f32],
    wtx2xyz: &[f32],
) -> (Vec<u32>, Vec<f32>) {
    let num_wtx = wtx2xyz.len() / 3;
    let mut wtx2elem = vec![0u32; num_wtx];
    let mut wtx2param = vec![0f32; num_wtx * 3];
    for i_wtx in 0..num_wtx {
        let query = arrayref::array_ref!(wtx2xyz, i_wtx * 3, 3);
        let mut best_dist_sq = f32::INFINITY;
        let mut best_elem = 0usize;
        let mut best_weights = [0f32; 3];
        nearest_elem_bvh(
            query,
            bvhnodes,
            bvhnode2aabb,
            elem2idx_offset,
            idx2vtx,
            vtx2xyz,
            0,
            &mut best_dist_sq,
            &mut best_elem,
            &mut best_weights,
        );
        wtx2elem[i_wtx] = best_elem as u32;
        wtx2param[i_wtx * 3..i_wtx * 3 + 3].copy_from_slice(&best_weights);
    }
    (wtx2elem, wtx2param)
}

pub fn vertices_on_face(
    i_elem: usize,
    i_face: usize,
    elem2idx_offset: &[usize],
    idx2vtx: &[usize],
) -> Vec<usize> {
    use crate::elem2elem::{PYRAMID_FACE2IDX, PYRAMID_IDX2NODE, TET_FACE2IDX, TET_IDX2NODE};
    let num_noel = elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem];
    let nofa2node = match num_noel {
        4 => &TET_IDX2NODE[TET_FACE2IDX[i_face]..TET_FACE2IDX[i_face + 1]],
        5 => &PYRAMID_IDX2NODE[PYRAMID_FACE2IDX[i_face]..PYRAMID_FACE2IDX[i_face + 1]],
        _ => {
            todo!()
        }
    };
    nofa2node
        .into_iter()
        .map(|i_node| idx2vtx[elem2idx_offset[i_elem] + i_node])
        .collect::<Vec<_>>()
}

pub fn elem2elem_with_vtx2elem(
    elem2idx_offset: &[usize],
    idx2vtx: &[usize],
    vtx2kdx_offset: &[usize],
    kdx2elem: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let num_elem = elem2idx_offset.len() - 1;
    let elem2jdx_offset = {
        let mut elem2jdx_offset = Vec::<usize>::with_capacity(num_elem + 1);
        elem2jdx_offset.push(0);
        for i_elem in 0..num_elem {
            let num_noel = elem2idx_offset[i_elem + 1] - elem2idx_offset[i_elem];
            let num_face = match num_noel {
                4 => 4, // tet
                5 => 5, // pyramid
                6 => 5, // prism
                8 => 6, // hex
                _ => {
                    todo!()
                }
            };
            elem2jdx_offset.push(elem2jdx_offset.last().unwrap() + num_face);
        }
        elem2jdx_offset
    };
    let &num_jdx = elem2jdx_offset.last().unwrap();
    let mut jdx2elem = vec![usize::MAX; num_jdx];
    for i_elem in 0..num_elem {
        let num_face_i = elem2jdx_offset[i_elem + 1] - elem2jdx_offset[i_elem];
        for i_face in 0..num_face_i {
            if jdx2elem[elem2jdx_offset[i_elem] + i_face] != usize::MAX {
                continue;
            }
            let mut vtxs_i = crate::polyhedron_mesh::vertices_on_face(
                i_elem,
                i_face,
                &elem2idx_offset,
                &idx2vtx,
            );
            let sum_i: usize = vtxs_i.iter().sum();
            vtxs_i.sort();
            let i0_vtx = vtxs_i[0];
            for &j_elem in &kdx2elem[vtx2kdx_offset[i0_vtx]..vtx2kdx_offset[i0_vtx + 1]] {
                if j_elem == i_elem { continue; }
                let num_face_j = elem2jdx_offset[j_elem + 1] - elem2jdx_offset[j_elem];
                for j_face in 0..num_face_j {
                    let mut vtxs_j = crate::polyhedron_mesh::vertices_on_face(
                        j_elem,
                        j_face,
                        &elem2idx_offset,
                        &idx2vtx,
                    );
                    if vtxs_i.len() != vtxs_j.len() {
                        continue;
                    }
                    if vtxs_j.iter().sum::<usize>() != sum_i {
                        continue;
                    }
                    vtxs_j.sort();
                    if vtxs_i != vtxs_j {
                        continue;
                    }
                    jdx2elem[elem2jdx_offset[i_elem] + i_face] = j_elem;
                    jdx2elem[elem2jdx_offset[j_elem] + j_face] = i_elem;
                    break;
                }
                if jdx2elem[elem2jdx_offset[i_elem] + i_face] != usize::MAX {
                    break;
                }
            }
        }
    }
    (elem2jdx_offset, jdx2elem)
}
