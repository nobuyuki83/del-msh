pub fn unique_for_sorted_array(idx2val: &[u32], idx2jdx: &mut [u32]) {
    let n = idx2val.len();
    assert_eq!(idx2jdx.len(), n);
    idx2jdx[0] = 0;
    for idx in 0..n - 1 {
        idx2jdx[idx + 1] = if idx2val[idx] == idx2val[idx + 1] {
            0
        } else {
            1
        };
    }
    for idx in 0..n - 1 {
        idx2jdx[idx + 1] += idx2jdx[idx];
    }
}
