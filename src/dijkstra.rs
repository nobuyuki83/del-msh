
struct Node
{
    ind: usize,
    dist: usize
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


/**
 * propergating from one element, finding the topologycal distance
 * @param ielm_ker
 * @param aElSuEl
 * @param nelem
 */
pub fn topological_distance_on_uniform_mesh(
    idx_elem_kernel: usize,
    elem2elem_adj: &[usize],
    num_elem: usize) -> Vec<usize>
{
    let num_edge = elem2elem_adj.len() / num_elem;
    assert_eq!(elem2elem_adj.len(), num_edge * num_elem);
    let mut elem2order = vec!(usize::MAX; num_elem);
    let mut elem2dist = vec!(usize::MAX; num_elem);
    elem2dist[idx_elem_kernel] = 0;
    let mut que = std::collections::BinaryHeap::<Node>::new();
    que.push(Node {ind: idx_elem_kernel, dist: 0});
    let mut count = 0;
    while !que.is_empty() {
        let top = que.pop().unwrap();
        let i_elm0 = top.ind;
        let i_dist0 = top.dist;
        if elem2order[i_elm0] != usize::MAX { continue; } // already fixed so this is not the shortest path
        elem2order[i_elm0] = count; // found shortest path
        count += 1;
        for i_edge in 0..num_edge {
            let i_elm1 = elem2elem_adj[i_elm0 * num_edge + i_edge];
            if i_elm1 == usize::MAX { continue; }
            let i_dist1 = i_dist0 + 1;
            if i_dist1 >= elem2dist[i_elm1] { continue; }
            elem2dist[i_elm1] = i_dist1; // Found the shortest path so far
            que.push(Node {ind: i_elm1, dist: i_dist1 }); // candidate of shortest path
        }
    }
    assert_eq!(count, num_elem);
    elem2dist
}