//! methods to compute graph distance on mesh

struct Node {
    ind: usize,
    dist: usize,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        other.dist.cmp(&self.dist)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Node {}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

/// propagating from one element, finding the topological distance
/// * `idx_elm_kernel` - index of element where distance is zero
/// * `elem2elem_adj` - index of adjacent element for each element
/// * `num_elem` - number of elements in the mesh
pub fn elem2dist_for_uniform_mesh<Index>(
    idx_elem_kernel: usize,
    elem2elem_adj: &[Index],
    num_elem: usize,
) -> Vec<usize>
where
    Index: num_traits::PrimInt + num_traits::AsPrimitive<usize>,
{
    let num_edge = elem2elem_adj.len() / num_elem;
    assert_eq!(elem2elem_adj.len(), num_edge * num_elem);
    let mut elem2order = vec![usize::MAX; num_elem];
    let mut elem2dist = vec![usize::MAX; num_elem];
    elem2dist[idx_elem_kernel] = 0;
    let mut que = std::collections::BinaryHeap::<Node>::new();
    que.push(Node {
        ind: idx_elem_kernel,
        dist: 0,
    });
    let mut count = 0;
    while !que.is_empty() {
        let top = que.pop().unwrap();
        let i_elm0 = top.ind;
        let i_dist0 = top.dist;
        if elem2order[i_elm0] != usize::MAX {
            continue;
        } // already fixed so this is not the shortest path
        elem2order[i_elm0] = count; // found shortest path
        count += 1;
        for i_edge in 0..num_edge {
            let i_elm1 = elem2elem_adj[i_elm0 * num_edge + i_edge];
            if i_elm1 == Index::max_value() {
                continue;
            }
            let i_dist1 = i_dist0 + 1;
            if i_dist1 >= elem2dist[i_elm1.as_()] {
                continue;
            }
            elem2dist[i_elm1.as_()] = i_dist1; // Found the shortest path so far
            que.push(Node {
                ind: i_elm1.as_(),
                dist: i_dist1,
            }); // candidate of shortest path
        }
    }
    assert!(count <= num_elem);
    elem2dist
}

/// propagating from one vertex, finding the topological distance
/// * `idx_vtx_kernel` - index of element where distance is zero
/// * `vtx2idx` - index of adjacent element for each element
/// * `idx2vtx` - index of adjacent element for each element
pub fn vtx2dist_for_vtx2vtx(
    idx_vtx_kernel: usize,
    vtx2idx: &[usize],
    idx2vtx: &[usize],
    max_dist: Option<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let num_vtx = vtx2idx.len() - 1;
    let mut vtx2order = vec![usize::MAX; num_vtx];
    let mut vtx2dist = vec![usize::MAX; num_vtx];
    vtx2dist[idx_vtx_kernel] = 0;
    let mut que = std::collections::BinaryHeap::<Node>::new();
    que.push(Node {
        ind: idx_vtx_kernel,
        dist: 0,
    });
    let mut count = 0;
    while !que.is_empty() {
        let top = que.pop().unwrap();
        let i_vtx0 = top.ind;
        let i_dist0 = top.dist;
        if vtx2order[i_vtx0] != usize::MAX {
            continue;
        } // already fixed so this is not the shortest path
        vtx2order[i_vtx0] = count; // found shortest path
        count += 1;
        for &j_vtx in &idx2vtx[vtx2idx[i_vtx0]..vtx2idx[i_vtx0 + 1]] {
            assert!(j_vtx < num_vtx);
            let i_dist1 = i_dist0 + 1;
            if i_dist1 >= vtx2dist[j_vtx] {
                continue;
            }
            vtx2dist[j_vtx] = i_dist1; // Found the shortest path so far
            que.push(Node {
                ind: j_vtx,
                dist: i_dist1,
            }); // candidate of shortest path
            if let Some(max_dist) = max_dist {
                if i_dist1 > max_dist {
                    return (vtx2dist, vtx2order);
                }
            }
        }
    }
    assert!(count <= num_vtx);
    (vtx2dist, vtx2order)
}

#[test]
fn test_vtx2dist_for_vtx2vtx() {
    let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::sphere_yup::<usize, f64>(1.0, 32, 32);
    let (vtx2idx, idx2vtx) =
        crate::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, vtx2xyz.len() / 3, false);
    let (vtx2dist, _) = vtx2dist_for_vtx2vtx(0, &vtx2idx, &idx2vtx, None);
    let &dist_max = vtx2dist.iter().max().unwrap();
    let vtx2rgb: Vec<_> = vtx2dist
        .iter()
        .flat_map(|&v| {
            let r = if v % 2 == 0 { 0. } else { 1. };
            let g = v as f32 / dist_max as f32;
            [r, g, 1.0 - g]
        })
        .collect();
    crate::io_obj::save_tri2vtx_vtx2xyz_vtx2rgb(
        "../target/test_vtx2dist_for_vtx2vtx.obj",
        &tri2vtx,
        &vtx2xyz,
        &vtx2rgb,
    )
    .unwrap();
}
