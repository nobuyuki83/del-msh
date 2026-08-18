/// merge to a block CSR data with diagonal matrix
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
pub fn csrdia<T, const BLKSIZE: usize, const NNODE: usize>(
    node2row: &[usize; NNODE],
    node2col: &[usize; NNODE],
    emat: &[[[T; BLKSIZE]; NNODE]; NNODE],
    row2idx: &[usize],
    idx2col: &[usize],
    row2val: &mut [T],
    idx2val: &mut [T],
    merge_buffer: &mut Vec<usize>,
) where
    T: std::ops::AddAssign + Copy,
{
    let num_blk = row2idx.len() - 1;
    assert_eq!(idx2val.len(), idx2col.len() * BLKSIZE);
    assert_eq!(row2val.len(), num_blk * BLKSIZE);
    merge_buffer.resize(num_blk, usize::MAX);
    let col2idx = merge_buffer;
    for inode in 0..NNODE {
        let i_row = node2row[inode];
        assert!(i_row < num_blk);
        for ij_idx in row2idx[i_row]..row2idx[i_row + 1] {
            assert!(ij_idx < idx2col.len());
            let j_col = idx2col[ij_idx];
            col2idx[j_col] = ij_idx;
        }
        for jnode in 0..NNODE {
            let j_col = node2col[jnode];
            assert!(j_col < num_blk);
            if i_row == j_col {
                // Marge Diagonal
                row2val[i_row] += emat[inode][jnode][0];
            } else {
                // Marge Non-Diagonal
                assert!(col2idx[j_col] < idx2col.len());
                let ij_idx = col2idx[j_col];
                assert_eq!(idx2col[ij_idx], j_col);
                idx2val[ij_idx] += emat[inode][jnode][0];
            }
        }
        for ij_idx in row2idx[i_row]..row2idx[i_row + 1] {
            assert!(ij_idx < idx2col.len());
            let j_col = idx2col[ij_idx];
            col2idx[j_col] = usize::MAX;
        }
    }
}

/// merge to block CSR data structure without diagonal matrix
pub fn blkcsr<T, const BLKSIZE: usize, const NNODE: usize>(
    node2row: &[usize; NNODE],
    node2col: &[usize; NNODE],
    emat: &[[[T; BLKSIZE]; NNODE]; NNODE],
    row2idx: &[usize],
    idx2col: &[usize],
    idx2val: &mut [T],
    merge_buffer: &mut Vec<usize>,
) where
    T: std::ops::AddAssign + Copy,
{
    let num_row = row2idx.len() - 1;
    merge_buffer.resize(num_row, usize::MAX);
    let col2idx = merge_buffer;
    for inode in 0..node2row.len() {
        let i_row = node2row[inode];
        assert!(i_row < num_row);
        for ij_idx in row2idx[i_row]..row2idx[i_row + 1] {
            assert!(ij_idx < idx2col.len());
            let j_col = idx2col[ij_idx];
            col2idx[j_col] = ij_idx;
        }
        for (jnode, &j_col) in node2col.iter().enumerate() {
            assert!(j_col < num_row);
            assert!(col2idx[j_col] < idx2col.len());
            let ij_idx = col2idx[j_col];
            assert_eq!(idx2col[ij_idx], j_col);
            for i in 0..BLKSIZE {
                idx2val[ij_idx * BLKSIZE + i] += emat[inode][jnode][i];
            }
        }
        for ij_idx in row2idx[i_row]..row2idx[i_row + 1] {
            assert!(ij_idx < idx2col.len());
            let j_col = idx2col[ij_idx];
            col2idx[j_col] = usize::MAX;
        }
    }
}
