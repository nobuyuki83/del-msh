//! Sparse Incomplete LU (ILU) factorization Preconditioner

pub struct Preconditioner<MAT> {
    pub num_blk: usize,
    pub row2idx: Vec<usize>,
    pub idx2col: Vec<usize>,
    pub row2idx_dia: Vec<usize>,
    pub idx2val: Vec<MAT>,
    pub row2val: Vec<MAT>,
}

impl<T, const BLKSIZE: usize> Default for Preconditioner<[T; BLKSIZE]>
where
    T: num_traits::Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const BLKSIZE: usize> Preconditioner<[T; BLKSIZE]>
where
    T: num_traits::Float,
{
    pub fn new() -> Self {
        Preconditioner {
            num_blk: 0,
            row2idx: vec![0],
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<[T; BLKSIZE]>::new(),
            row2val: Vec::<[T; BLKSIZE]>::new(),
            row2idx_dia: Vec::<usize>::new(),
        }
    }

    /// initialize non-zero pattern as ILU-0, i.e., same as the original matrix
    pub fn initialize_ilu0(
        &mut self,
        a: &crate::sparse_solver::sparse_square::MatrixOwned<[T; BLKSIZE]>,
    ) {
        let num_row = a.num_blk;
        self.num_blk = num_row;
        self.row2idx = a.row2idx.clone();
        self.idx2col = a.idx2col.clone();
        self.idx2val = vec![[T::zero(); BLKSIZE]; a.idx2col.len()];
        self.row2val = vec![[T::zero(); BLKSIZE]; num_row];
        self.row2idx_dia = vec![0_usize; num_row];
        // ---------------
        // sort idx2col
        for i_row in 0..num_row {
            let idx0 = self.row2idx[i_row];
            let idx1 = self.row2idx[i_row + 1];
            self.idx2col[idx0..idx1].sort();
        }
        // set row2idx_dia
        for i_row in 0..num_row {
            self.row2idx_dia[i_row] = self.row2idx[i_row + 1];
            for ij_idx in self.row2idx[i_row]..self.row2idx[i_row + 1] {
                assert!(ij_idx < self.idx2col.len());
                let j_col = self.idx2col[ij_idx];
                assert!(j_col < num_row);
                if j_col > i_row {
                    self.row2idx_dia[i_row] = ij_idx;
                    break;
                }
            }
        }
    }

    pub fn initialize_iluk(
        &mut self,
        a: &crate::sparse_solver::sparse_square::MatrixOwned<[T; BLKSIZE]>,
        k: usize,
    ) {
        (self.row2idx, self.idx2col, self.row2idx_dia) = symbolic_iluk(&a.row2idx, &a.idx2col, k);
        self.num_blk = a.num_blk;
        self.idx2val = vec![[T::zero(); BLKSIZE]; self.idx2col.len()];
        self.row2val = vec![[T::zero(); BLKSIZE]; a.num_blk];
    }

    /// initialize non-zero pattern as full matrix
    pub fn initialize_full(&mut self, num_row: usize) {
        self.num_blk = num_row;
        self.row2idx.resize(num_row + 1, 0_usize);
        for i_row in 0..num_row + 1 {
            self.row2idx[i_row] = i_row * (num_row - 1);
        }
        self.idx2col = vec![0_usize; num_row * (num_row - 1)];
        for i_row in 0..num_row {
            for j_col in 0..i_row {
                self.idx2col[i_row * (num_row - 1) + j_col] = j_col;
            }
            for j_col in i_row + 1..num_row {
                self.idx2col[i_row * (num_row - 1) + j_col - 1] = j_col;
            }
        }
        self.row2idx_dia = (0..num_row).map(|i| i * num_row).collect();
        self.idx2val = vec![[T::zero(); BLKSIZE]; self.idx2col.len()];
        self.row2val = vec![[T::zero(); BLKSIZE]; num_row];
    }
}

pub fn copy_value<T, const BLKSIZE: usize>(
    ilu: &mut Preconditioner<[T; BLKSIZE]>,
    a: &crate::sparse_solver::sparse_square::MatrixOwned<[T; BLKSIZE]>,
) where
    T: num_traits::Zero + Copy,
{
    let num_row = ilu.num_blk;
    assert_eq!(a.num_blk, num_row);
    let mut col2idx = vec![usize::MAX; num_row];
    // copy diagonal value
    crate::sparse_solver::slice_of_array::copy(&mut ilu.row2val, &a.row2val);
    // copy off-diagonal values
    ilu.idx2val
        .iter_mut()
        .for_each(|v| *v = [T::zero(); BLKSIZE]);
    for i_row in 0..num_row {
        for ij_idx0 in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx0 < ilu.idx2col.len());
            let j_col0 = ilu.idx2col[ij_idx0];
            assert!(j_col0 < num_row);
            col2idx[j_col0] = ij_idx0;
        }
        for a_ij_idx in a.row2idx[i_row]..a.row2idx[i_row + 1] {
            assert!(a_ij_idx < a.idx2col.len());
            let j_col0 = a.idx2col[a_ij_idx];
            assert!(j_col0 < num_row);
            let ij_idx = col2idx[j_col0];
            if ij_idx != usize::MAX {
                ilu.idx2val[ij_idx] = a.idx2val[a_ij_idx];
            } else {
                panic!();
            }
        }
        for ij_idx0 in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx0 < ilu.idx2col.len());
            let j_col0 = ilu.idx2col[ij_idx0];
            assert!(j_col0 < num_row);
            col2idx[j_col0] = usize::MAX;
        }
    }
}

pub fn decompose<T, const N: usize, const NN: usize>(
    ilu: &mut Preconditioner<[T; NN]>,
) -> Result<(), String>
where
    T: num_traits::Float + std::fmt::Debug,
{
    use del_geo_core::matn_col_major;
    let num_row = ilu.num_blk;
    let mut col2idx = vec![usize::MAX; num_row];
    for i_row in 0..num_row {
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < num_row);
            col2idx[j_col] = ij_idx;
        }
        // [L] * [D^-1*U]
        for ik_idx in ilu.row2idx[i_row]..ilu.row2idx_dia[i_row] {
            let k_colrow = ilu.idx2col[ik_idx];
            assert!(k_colrow < num_row);
            let ik_val = ilu.idx2val[ik_idx];
            for kj_idx in ilu.row2idx_dia[k_colrow]..ilu.row2idx[k_colrow + 1] {
                let j_col = ilu.idx2col[kj_idx];
                assert!(j_col < num_row);
                let kj_val = ilu.idx2val[kj_idx];
                let ikkj = matn_col_major::mult_mat_col_major::<T, N, NN>(&ik_val, &kj_val);
                if j_col != i_row {
                    let ij_idx = col2idx[j_col];
                    if ij_idx == usize::MAX {
                        continue;
                    }
                    matn_col_major::sub_in_place(&mut ilu.idx2val[ij_idx], &ikkj);
                } else {
                    matn_col_major::sub_in_place(&mut ilu.row2val[i_row], &ikkj);
                }
            }
        }
        // invserse diagonal
        // dbg!(&ilu.row2val[i_row]);
        let res = matn_col_major::try_inverse::<T, N, NN>(&ilu.row2val[i_row]);
        if res.is_none() {
            return Err("frac_failure".to_string());
        }
        ilu.row2val[i_row] = res.unwrap();
        // [U] = [1/D][U]
        for ij_idx in ilu.row2idx_dia[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let a = matn_col_major::mult_mat_col_major::<T, N, NN>(
                &ilu.row2val[i_row],
                &ilu.idx2val[ij_idx],
            );
            ilu.idx2val[ij_idx] = a;
        }
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < num_row);
            col2idx[j_col] = usize::MAX;
        }
    } // end iblk
    Ok(())
}

pub fn solve_preconditioning_vec<T, const NDIMVAL: usize, const BLKSIZE: usize>(
    vec: &mut [[T; NDIMVAL]],
    ilu: &Preconditioner<[T; BLKSIZE]>,
) where
    T: num_traits::Float + std::fmt::Debug,
{
    use del_geo_core::matn_col_major;
    use del_geo_core::vecn::VecN;
    assert_eq!(vec.len(), ilu.row2val.len());
    // forward
    let num_row = ilu.num_blk;
    for i_row in 0..num_row {
        let mut t = vec[i_row];
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx_dia[i_row] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < i_row);
            let v = matn_col_major::mult_vec(&ilu.idx2val[ij_idx], &vec[j_col]);
            t.sub_in_place(&v); // jblk0!=iblk
        }
        vec[i_row] = matn_col_major::mult_vec(&ilu.row2val[i_row], &t);
    }
    // -----
    // backward
    for i_row in (0..num_row).rev() {
        let mut t = vec[i_row];
        assert!(i_row < num_row);
        for ij_idx in ilu.row2idx_dia[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col > i_row && j_col < num_row);
            let v = matn_col_major::mult_vec(&ilu.idx2val[ij_idx], &vec[j_col]);
            t.sub_in_place(&v); // jblk0!=iblk
        }
        vec[i_row] = t;
    }
}

fn symbolic_iluk(
    a_row2idx: &[usize],
    a_idx2col: &[usize],
    lev_fill: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut a_idx2col: Vec<usize> = a_idx2col.to_vec();
    let num_row = a_row2idx.len() - 1;
    for i_row in 0..num_row {
        let idx0 = a_row2idx[i_row];
        let idx1 = a_row2idx[i_row + 1];
        a_idx2col[idx0..idx1].sort();
    }
    let mut row2idx = vec![0_usize; num_row + 1];
    let mut row2idx_dia = vec![0; num_row];

    let mut idx2collev = Vec::<[usize; 2]>::with_capacity(a_idx2col.len() * 4);

    for i_row in 0..num_row {
        let mut col2lev = std::collections::BTreeMap::<usize, usize>::new();
        let mut que0 = std::collections::BTreeSet::<usize>::new();
        for &j_col in &a_idx2col[a_row2idx[i_row]..a_row2idx[i_row + 1]] {
            col2lev.insert(j_col, 0);
            if j_col < i_row {
                que0.insert(j_col);
            }
        }
        loop {
            // loop while que is not empty
            let k_colrow = match que0.iter().next() {
                None => break,
                Some(x) => *x,
            };
            que0.remove(&k_colrow);
            assert!(k_colrow < i_row);
            let ik_lev0 = *col2lev.get(&k_colrow).unwrap();
            if ik_lev0 + 1 > lev_fill {
                continue;
            } // move next
            #[allow(clippy::needless_range_loop)]
            for kj_idx in row2idx_dia[k_colrow]..row2idx[k_colrow + 1] {
                let kj_lev0 = idx2collev[kj_idx][1];
                if kj_lev0 + 1 > lev_fill {
                    continue;
                }
                let j_col = idx2collev[kj_idx][0];
                assert!(j_col > k_colrow && j_col < num_row);
                if j_col == i_row {
                    continue;
                } // already filled-in on the diagonal
                let max_lev0 = if ik_lev0 > kj_lev0 { ik_lev0 } else { kj_lev0 };
                if j_col < i_row {
                    que0.insert(j_col);
                }
                match col2lev.get_mut(&j_col) {
                    None => {
                        col2lev.insert(j_col, max_lev0 + 1);
                    }
                    Some(lev) => {
                        *lev = if *lev < max_lev0 + 1 {
                            *lev
                        } else {
                            max_lev0 + 1
                        };
                    }
                }
            }
        }
        {
            // finalize this row
            let ij_idx1 = row2idx[i_row] + col2lev.len();
            idx2collev.reserve(row2idx[i_row] + col2lev.len());
            for (&col, &lev) in col2lev.iter() {
                idx2collev.push([col, lev]);
            }
            row2idx[i_row + 1] = ij_idx1;
            // set row2idx_dia
            row2idx_dia[i_row] = ij_idx1;
            #[allow(clippy::needless_range_loop)]
            for ij_idx1 in row2idx[i_row]..row2idx[i_row + 1] {
                let j_col = idx2collev[ij_idx1][0];
                if j_col > i_row {
                    row2idx_dia[i_row] = ij_idx1;
                    break;
                }
            }
        }
    }

    let mut idx2col = vec![0_usize; idx2collev.len()];
    for icrs in 0..idx2col.len() {
        idx2col[icrs] = idx2collev[icrs][0];
    }
    (row2idx, idx2col, row2idx_dia)
}

/*
impl<MAT> Preconditioner<MAT>
where
    MAT: 'static + Copy + num_traits::Zero,
    f32: AsPrimitive<MAT>,
{
    pub fn new() -> Self {
        Preconditioner {
            num_blk: 0,
            row2idx: vec![0],
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<MAT>::new(),
            row2val: Vec::<MAT>::new(),
            row2idx_dia: Vec::<usize>::new(),
        }
    }

    pub fn clone(&self) -> Self {
        Preconditioner {
            num_blk: self.num_blk,
            row2idx: self.row2idx.clone(),
            idx2col: self.idx2col.clone(),
            idx2val: self.idx2val.clone(),
            row2val: self.row2val.clone(),
            row2idx_dia: self.row2idx_dia.clone(),
        }
    }



    /// initialize non-zero pattern as full matrix
    pub fn initialize_full(&mut self, num_row: usize) {
        self.num_blk = num_row;
        self.row2idx.resize(num_row + 1, 0_usize);
        for i_row in 0..num_row + 1 {
            self.row2idx[i_row] = i_row * (num_row - 1);
        }
        self.idx2col = vec![0_usize; num_row * (num_row - 1)];
        for i_row in 0..num_row {
            for j_col in 0..i_row {
                self.idx2col[i_row * (num_row - 1) + j_col] = j_col;
            }
            for j_col in i_row + 1..num_row {
                self.idx2col[i_row * (num_row - 1) + j_col - 1] = j_col;
            }
        }
        self.row2idx_dia = (0..num_row).map(|i| i * num_row).collect();
        self.idx2val = vec![MAT::zero(); self.idx2col.len()];
        self.row2val = vec![MAT::zero(); num_row];
    }

    /// initialize non-zero pattern with ILU-k symbolic factorization
    /// * `fill-level` - fill-in level
    pub fn initialize_iluk(&mut self, a: &crate::sparse_square::Matrix<MAT>, lev_fill: usize) {
        if lev_fill == 0 {
            self.initialize_ilu0(a);
            return;
        }
        (self.row2idx, self.idx2col, self.row2idx_dia) =
            symbolic_iluk(&a.row2idx, a.idx2col.clone(), lev_fill);
        self.num_blk = self.row2idx.len() - 1;
        self.idx2val.resize(self.idx2col.len(), MAT::zero());
        self.row2val.resize(self.num_blk, MAT::zero());
    }
}

pub fn copy_value<MAT>(ilu: &mut Preconditioner<MAT>, a: &crate::sparse_square::Matrix<MAT>)
where
    MAT: num_traits::Zero + Copy,
{
    let num_row = ilu.num_blk;
    assert_eq!(a.num_blk, num_row);
    let mut col2idx = vec![usize::MAX; num_row];
    // copy diagonal value
    ilu.row2val =  &a.row2val;
    // copy off-diagonal values
    ilu.idx2val.iter_mut().for_each(|v| v.set_zero());
    for i_row in 0..num_row {
        for ij_idx0 in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx0 < ilu.idx2col.len());
            let j_col0 = ilu.idx2col[ij_idx0];
            assert!(j_col0 < num_row);
            col2idx[j_col0] = ij_idx0;
        }
        for a_ij_idx in a.row2idx[i_row]..a.row2idx[i_row + 1] {
            assert!(a_ij_idx < a.idx2col.len());
            let j_col0 = a.idx2col[a_ij_idx];
            assert!(j_col0 < num_row);
            let ij_idx = col2idx[j_col0];
            if ij_idx != usize::MAX {
                ilu.idx2val[ij_idx] = a.idx2val[a_ij_idx];
            } else {
                panic!();
            }
        }
        for ij_idx0 in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx0 < ilu.idx2col.len());
            let j_col0 = ilu.idx2col[ij_idx0];
            assert!(j_col0 < num_row);
            col2idx[j_col0] = usize::MAX;
        }
    }
}

pub fn decompose<T>(ilu: &mut Preconditioner<T>)
where
    T: 'static + Copy + std::ops::Mul<Output = T> + std::ops::SubAssign + std::ops::Div<Output = T>,
    f32: AsPrimitive<T>,
{
    let num_row = ilu.num_blk;
    let mut col2idx = vec![usize::MAX; num_row];
    for i_row in 0..num_row {
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < num_row);
            col2idx[j_col] = ij_idx;
        }
        // [L] * [D^-1*U]
        for ik_idx in ilu.row2idx[i_row]..ilu.row2idx_dia[i_row] {
            let k_colrow = ilu.idx2col[ik_idx];
            assert!(k_colrow < num_row);
            let ik_val = ilu.idx2val[ik_idx];
            for kj_idx in ilu.row2idx_dia[k_colrow]..ilu.row2idx[k_colrow + 1] {
                let j_col = ilu.idx2col[kj_idx];
                assert!(j_col < num_row);
                let kj_val = ilu.idx2val[kj_idx];
                if j_col != i_row {
                    let ij_idx = col2idx[j_col];
                    if ij_idx == usize::MAX {
                        continue;
                    }
                    ilu.idx2val[ij_idx] -= ik_val * kj_val;
                } else {
                    ilu.row2val[i_row] -= ik_val * kj_val;
                }
            }
        }
        // invserse diagonal
        ilu.row2val[i_row] = 1_f32.as_() / ilu.row2val[i_row];
        // [U] = [1/D][U]
        for ij_idx in ilu.row2idx_dia[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            ilu.idx2val[ij_idx] = ilu.idx2val[ij_idx] * ilu.row2val[i_row];
        }
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < num_row);
            col2idx[j_col] = usize::MAX;
        }
    } // end iblk
}

pub fn solve_preconditioning_vec<T>(vec: &mut Vec<T>, ilu: &Preconditioner<T>)
where
    T: Copy + std::ops::Mul<Output = T> + std::ops::SubAssign,
{
    assert_eq!(vec.len(), ilu.row2val.len());
    // forward
    let num_row = ilu.num_blk;
    for i_row in 0..num_row {
        for ij_idx in ilu.row2idx[i_row]..ilu.row2idx_dia[i_row] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col < i_row);
            let v = ilu.idx2val[ij_idx] * vec[j_col];
            vec[i_row] -= v; // jblk0!=iblk
        }
        vec[i_row] = ilu.row2val[i_row] * vec[i_row];
    }
    // -----
    // backward
    for i_row in (0..num_row).rev() {
        assert!(i_row < num_row);
        for ij_idx in ilu.row2idx_dia[i_row]..ilu.row2idx[i_row + 1] {
            assert!(ij_idx < ilu.idx2col.len());
            let j_col = ilu.idx2col[ij_idx];
            assert!(j_col > i_row && j_col < num_row);
            let v = ilu.idx2val[ij_idx] * vec[j_col];
            vec[i_row] -= v; // jblk0!=iblk
        }
    }
}


*/
