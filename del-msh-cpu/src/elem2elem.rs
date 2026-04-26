//! methods that generate the elements adjacent to an element
use num_traits::AsPrimitive;

pub const TET_FACE2IDX: [usize; 5] = [0, 3, 6, 9, 12];
pub const TET_IDX2NODE: [usize; 12] = [1, 2, 3, 2, 0, 3, 0, 1, 3, 0, 2, 1];

pub const EDGE_FACE2IDX: [usize; 3] = [0, 1, 2];
pub const EDGE_IDX2NODE: [usize; 2] = [1, 0];

pub const TRI_FACE2IDX: [usize; 4] = [0, 2, 4, 6];
pub const TRI_IDX2NODE: [usize; 6] = [1, 2, 2, 0, 0, 1];

pub const PYRAMID_FACE2IDX: [usize; 6] = [0, 4, 7, 10, 13, 16];
pub const PYRAMID_IDX2NODE: [usize; 16] = [0, 3, 2, 1, 0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];

pub fn face2node_of_polygon_element(num_node: usize) -> (Vec<usize>, Vec<usize>) {
    let mut face2idx = vec![0; num_node + 1];
    let mut idx2node = vec![0; num_node * 2];
    for i_edge in 0..num_node {
        face2idx[i_edge + 1] = (i_edge + 1) * 2;
        idx2node[i_edge * 2] = i_edge;
        idx2node[i_edge * 2 + 1] = (i_edge + 1) % num_node;
    }
    (face2idx, idx2node)
}

pub fn face2node_of_simplex_element(num_node: usize) -> (Vec<usize>, Vec<usize>) {
    match num_node {
        2 => (EDGE_FACE2IDX.to_vec(), EDGE_IDX2NODE.to_vec()),
        3 => (TRI_FACE2IDX.to_vec(), TRI_IDX2NODE.to_vec()),
        4 => (TET_FACE2IDX.to_vec(), TET_IDX2NODE.to_vec()),
        _ => {
            panic!()
        }
    }
}

/// element adjacency of uniform mesh
/// * `elem2vtx` - vertex index of elements
/// * `num_node` - number of nodes par element
/// * `vtx2elem_idx` - jagged array index of element surrounding point
/// * `vtx2elem` - jagged array value of  element surrounding point
///
///  triangle: `face2jdx` = \[0,2,4,6]; `jdx2node` = \[1,2,2,0,0,1];
pub fn from_uniform_mesh_with_vtx2elem<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    vtx2idx: &[Index],
    idx2elem: &[Index],
    face2jdx: &[usize],
    jdx2node: &[usize],
) -> Vec<Index>
where
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: num_traits::AsPrimitive<Index>,
{
    assert!(!vtx2idx.is_empty());
    let num_vtx = vtx2idx.len() - 1;
    let num_face_par_elem = face2jdx.len() - 1;
    let num_max_node_on_face = {
        let mut n0 = 0_usize;
        for i_face in 0..num_face_par_elem {
            let nno = face2jdx[i_face + 1] - face2jdx[i_face];
            n0 = if nno > n0 { nno } else { n0 }
        }
        n0
    };

    let num_elem = elem2vtx.len() / num_node;
    let mut elem2elem = vec![Index::max_value(); num_elem * num_face_par_elem];

    let mut vtx2flag = vec![0; num_vtx]; // vertex index -> flag
    let mut jdx2vtx = vec![0; num_max_node_on_face]; // face node index -> vertex index
    for i_elem in 0..num_elem {
        for i_face in 0..num_face_par_elem {
            for jdx0 in 0..face2jdx[i_face + 1] - face2jdx[i_face] {
                let i_node0 = jdx2node[jdx0 + face2jdx[i_face]];
                assert!(i_node0 < num_node);
                let i_vtx: usize = elem2vtx[i_elem * num_node + i_node0].as_();
                assert!(i_vtx < num_vtx);
                jdx2vtx[jdx0] = i_vtx;
                vtx2flag[i_vtx] = 1;
            }
            let i_vtx0 = jdx2vtx[0];
            let mut flag0 = false;
            for &j_elem0 in &idx2elem[vtx2idx[i_vtx0].as_()..vtx2idx[i_vtx0 + 1].as_()] {
                let j_elem0: usize = j_elem0.as_();
                if j_elem0 == i_elem {
                    continue;
                }
                for j_face in 0..num_face_par_elem {
                    flag0 = true;
                    for &j_node0 in &jdx2node[face2jdx[j_face]..face2jdx[j_face + 1]] {
                        let j_vtx0: usize = elem2vtx[j_elem0 * num_node + j_node0].as_();
                        if vtx2flag[j_vtx0] == 0 {
                            flag0 = false;
                            break;
                        }
                    }
                    if flag0 {
                        elem2elem[i_elem * num_face_par_elem + i_face] = j_elem0.as_();
                        break;
                    }
                }
                if flag0 {
                    break;
                }
            }
            if !flag0 {
                elem2elem[i_elem * num_face_par_elem + i_face] = Index::max_value();
            }
            for ifano in 0..face2jdx[i_face + 1] - face2jdx[i_face] {
                vtx2flag[jdx2vtx[ifano]] = 0;
            }
        }
    }
    elem2elem
}

/// element surrounding element
/// * `elem2vtx` - vertex index of elements
/// * `num_node` - number of nodes par element
/// * `num_vtx` - number of vertices
///
///  triangle: face2idx = \[0,2,4,6]; idx2node = \[1,2,2,0,0,1];
pub fn from_uniform_mesh<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    face2idx: &[usize],
    idx2node: &[usize],
    num_vtx: usize,
) -> Vec<Index>
where
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: num_traits::AsPrimitive<Index>,
{
    let vtx2elem = crate::vtx2elem::from_uniform_mesh(elem2vtx, num_node, num_vtx);
    from_uniform_mesh_with_vtx2elem(
        elem2vtx,
        num_node,
        &vtx2elem.0,
        &vtx2elem.1,
        face2idx,
        idx2node,
    )
}

pub fn from_polygon_mesh_with_vtx2elem(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    vtx2jdx: &[usize],
    jdx2elem: &[usize],
) -> Vec<usize> {
    assert!(!vtx2jdx.is_empty());
    let num_elem = elem2idx.len() - 1;
    let mut idx2elem = vec![usize::MAX; idx2vtx.len()];
    for i_elem in 0..num_elem {
        let num_edge_in_i_elem = elem2idx[i_elem + 1] - elem2idx[i_elem];
        for i_edge in 0..num_edge_in_i_elem {
            let i_edge2vtx = [
                idx2vtx[elem2idx[i_elem] + i_edge],
                idx2vtx[elem2idx[i_elem] + (i_edge + 1) % num_edge_in_i_elem],
            ];
            let i_vtx0 = i_edge2vtx[0];
            for &j_elem0 in &jdx2elem[vtx2jdx[i_vtx0]..vtx2jdx[i_vtx0 + 1]] {
                if j_elem0 == i_elem {
                    continue;
                }
                let num_edge_in_j_elem0 = elem2idx[j_elem0 + 1] - elem2idx[j_elem0];
                for j_edge in 0..num_edge_in_j_elem0 {
                    let j_edge2vtx = [
                        idx2vtx[elem2idx[j_elem0] + j_edge],
                        idx2vtx[elem2idx[j_elem0] + (j_edge + 1) % num_edge_in_j_elem0],
                    ];
                    if i_edge2vtx[0] != j_edge2vtx[1] || i_edge2vtx[1] != j_edge2vtx[0] {
                        continue;
                    }
                    idx2elem[elem2idx[i_elem] + i_edge] = j_elem0;
                    break;
                }
                if idx2elem[elem2idx[i_elem] + i_edge] != usize::MAX {
                    break;
                }
            }
        }
    }
    idx2elem
}

pub fn from_polygon_mesh(elem2idx: &[usize], idx2vtx: &[usize], num_vtx: usize) -> Vec<usize> {
    let vtx2elem = crate::vtx2elem::from_polygon_mesh(elem2idx, idx2vtx, num_vtx);
    from_polygon_mesh_with_vtx2elem(elem2idx, idx2vtx, &vtx2elem.0, &vtx2elem.1)
}

/// Extract the boundary surface mesh from a uniform volumetric mesh.
///
/// A face is on the boundary when its `elem2elem` entry equals `Index::max_value()`.
/// Returns a flat array of vertex indices for the boundary faces (uniform surface mesh).
/// The number of nodes per boundary face is `face2idx[1] - face2idx[0]`.
///
/// # Arguments
/// * `elem2vtx` - vertex indices of elements, length `num_elem * num_node`
/// * `num_node` - number of nodes per element (e.g. 4 for tets)
/// * `elem2elem` - element adjacency array, length `num_elem * num_face_per_elem`;
///   boundary faces have value `Index::max_value()`
/// * `face2idx` - CSR offsets into `idx2node` for each face of an element
/// * `idx2node` - local node indices on each face
pub fn extract_boundary_mesh_for_uniform_mesh<Index>(
    elem2vtx: &[Index],
    num_node: usize,
    elem2elem: &[Index],
    face2idx: &[usize],
    idx2node: &[usize],
) -> Vec<Index>
where
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
    usize: num_traits::AsPrimitive<Index>,
{
    let num_face_per_elem = face2idx.len() - 1;
    let num_elem = elem2vtx.len() / num_node;
    let mut bnd_face2vtx = Vec::<Index>::new();
    for i_elem in 0..num_elem {
        for i_face in 0..num_face_per_elem {
            if elem2elem[i_elem * num_face_per_elem + i_face] != Index::max_value() {
                continue;
            }
            for jdx in face2idx[i_face]..face2idx[i_face + 1] {
                let i_node = idx2node[jdx];
                bnd_face2vtx.push(elem2vtx[i_elem * num_node + i_node]);
            }
        }
    }
    bnd_face2vtx
}
