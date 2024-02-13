// topology
pub mod vtx2elem;
pub mod vtx2vtx;
pub mod elem2elem;
pub mod edge2vtx;
pub mod tri2vtx;
pub mod unify_index;

// functions specific to type of mesh
pub mod map_idx;
pub mod polyloop;
pub mod polyloop2;
pub mod polyloop3;
pub mod polyline;
pub mod polyline3;
pub mod trimesh2;
pub mod trimesh3;
pub mod trimesh3_search_bruteforce;
pub mod trimesh3_primitive;
pub mod quadmesh;
pub mod vtx2xyz;

// misc functions general to mesh type
pub mod elem2group;
pub mod elem2center;
pub mod sampling;
pub mod transform;
pub mod extract;
pub mod unindex;
pub mod dijkstra;

// io
pub mod io_obj;
pub mod io_off;
pub mod io_vtk;
pub mod io_nas;

// search
pub mod kdtree2;
pub mod bvh3;
pub mod bvh3_topology_morton;
pub mod bvh;
pub mod bvh3_topology_topdown;

// self intersection
pub mod trimesh3_intersection;
pub mod trimesh3_intersection_time;
pub mod trimesh3_proximity;
pub mod trimesh3_move_avoid_intersection;
pub mod mesh_laplacian;


pub fn merge<T>(
    node2row: &[usize],
    node2col: &[usize],
    emat: &[T],
    row2idx: &[usize],
    idx2col: &[usize],
    row2val: &mut [T],
    idx2val: &mut [T],
    merge_buffer: &mut Vec<usize>)
where T: std::ops::AddAssign + Copy
{
    let num_blk = row2idx.len() - 1;
    assert_eq!(emat.len(), node2row.len() * node2col.len());
    merge_buffer.resize(num_blk, usize::MAX);
    let col2idx = merge_buffer;
    for inode in 0..node2row.len() {
        let i_row = node2row[inode];
        assert!(i_row < num_blk);
        for ij_idx in row2idx[i_row]..row2idx[i_row + 1] {
            assert!(ij_idx < idx2col.len());
            let j_col = idx2col[ij_idx];
            col2idx[j_col] = ij_idx;
        }
        for jnode in 0..node2col.len() {
            let j_col = node2col[jnode];
            assert!(j_col < num_blk);
            if i_row == j_col {  // Marge Diagonal
                row2val[i_row] += emat[inode * node2col.len() + jnode];
            } else {  // Marge Non-Diagonal
                assert!(col2idx[j_col] < idx2col.len());
                let ij_idx = col2idx[j_col];
                assert_eq!(idx2col[ij_idx], j_col);
                idx2val[ij_idx] += emat[inode * node2col.len() + jnode];
            }
        }
        for ij_idx in row2idx[i_row]..row2idx[i_row + 1] {
            assert!(ij_idx < idx2col.len());
            let j_col = idx2col[ij_idx];
            col2idx[j_col] = usize::MAX;
        }
    }
}