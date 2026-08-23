//! sparse matrix class and functions

pub fn set_fixed_bc<T, const NDIMVAL: usize, const BLKSIZE: usize>(
    val_dia: T,
    bc_flag: &[[i32; NDIMVAL]],
    row2val: &mut [[T; BLKSIZE]],
    idx2val: &mut [[T; BLKSIZE]],
    row2idx: &[usize],
    idx2col: &[usize],
) where
    T: num_traits::Float,
{
    let num_blk = bc_flag.len();
    assert_eq!(bc_flag.len(), row2val.len());
    for i_blk in 0..num_blk {
        // set diagonal
        for i_dim in 0..NDIMVAL {
            if bc_flag[i_blk][i_dim] == 0 {
                continue;
            };
            for j_dim in 0..NDIMVAL {
                row2val[i_blk][i_dim + NDIMVAL * j_dim] = T::zero();
                row2val[i_blk][j_dim + NDIMVAL * i_dim] = T::zero();
            }
            row2val[i_blk][i_dim + NDIMVAL * i_dim] = val_dia;
        }
    }
    //
    assert_eq!(bc_flag.len(), num_blk);
    for i_blk in 0..num_blk {
        // set row
        #[allow(clippy::needless_range_loop)]
        for idx in row2idx[i_blk]..row2idx[i_blk + 1] {
            for i_dim in 0..NDIMVAL {
                if bc_flag[i_blk][i_dim] == 0 {
                    continue;
                };
                for j_dim in 0..NDIMVAL {
                    idx2val[idx][i_dim + NDIMVAL * j_dim] = T::zero();
                }
            }
        }
    }
    //
    for idx in 0..idx2col.len() {
        let j_blk1 = idx2col[idx];
        for j_dim in 0..NDIMVAL {
            if bc_flag[j_blk1][j_dim] == 0 {
                continue;
            };
            for i_dim in 0..NDIMVAL {
                idx2val[idx][i_dim + NDIMVAL * j_dim] = T::zero();
            }
        }
    }
}

pub fn set_fix_dof_to_rhs_vector<T, const NDIMVAL: usize>(
    blk2rhs: &mut [[T; NDIMVAL]],
    blk2isfix: &[[i32; NDIMVAL]],
) where
    T: num_traits::Float,
{
    let num_vtx = blk2rhs.len();
    for i_vtx in 0..num_vtx {
        for i_dof in 0..NDIMVAL {
            if blk2isfix[i_vtx][i_dof] == 0 {
                continue;
            }
            blk2rhs[i_vtx][i_dof] = T::zero();
        }
    }
}

/// sparse matrix class
/// Compressed Row Storage (CRS) data structure
/// * `num_blk` - number of row and col blocks
pub struct MatrixOwned<MAT> {
    pub num_blk: usize,
    pub row2idx: Vec<usize>,
    pub idx2col: Vec<usize>,
    pub idx2val: Vec<MAT>,
    pub row2val: Vec<MAT>,
}

impl<T, const BLKSIZE: usize> Default for MatrixOwned<[T; BLKSIZE]>
where
    T: num_traits::Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const BLKSIZE: usize> MatrixOwned<[T; BLKSIZE]>
where
    T: num_traits::Float,
{
    pub fn new() -> Self {
        MatrixOwned {
            num_blk: 0,
            row2idx: vec![0],
            idx2col: Vec::<usize>::new(),
            idx2val: Vec::<[T; BLKSIZE]>::new(),
            row2val: Vec::<[T; BLKSIZE]>::new(),
        }
    }

    pub fn as_ref_mut(&mut self) -> MatrixRefMut<'_, T, BLKSIZE> {
        MatrixRefMut {
            num_blk: self.num_blk,
            row2idx: &self.row2idx,
            idx2col: &self.idx2col,
            idx2val: &mut self.idx2val,
            row2val: &mut self.row2val,
        }
    }

    pub fn as_ref(&self) -> MatrixRef<'_, T, BLKSIZE> {
        MatrixRef {
            num_blk: self.num_blk,
            row2idx: &self.row2idx,
            idx2col: &self.idx2col,
            idx2val: &self.idx2val,
            row2val: &self.row2val,
        }
    }

    pub fn from_vtx2vtx(vtx2idx: &[usize], idx2vtx: &[usize]) -> Self {
        let num_blk = vtx2idx.len() - 1;
        let num_idx = vtx2idx[num_blk];
        Self {
            num_blk,
            row2idx: vtx2idx.to_vec(),
            idx2col: idx2vtx.to_vec(),
            idx2val: vec![[T::zero(); BLKSIZE]; num_idx],
            row2val: vec![[T::zero(); BLKSIZE]; num_blk],
        }
    }

    pub fn set_fixed<const NDIMVAL: usize>(&mut self, val_dia: T, blk2isfix: &[[i32; NDIMVAL]]) {
        set_fixed_bc(
            val_dia,
            blk2isfix,
            &mut self.row2val,
            &mut self.idx2val,
            &self.row2idx,
            &self.idx2col,
        );
    }

    /// generalized matrix-vector multiplication
    /// where matrix is sparse (not block) matrix
    /// `{y_vec} <- \alpha * [a_mat] * {x_vec} + \beta * {y_vec}`
    pub fn mult_vec<const NDIMVAL: usize>(
        &self,
        y_vec: &mut [[T; NDIMVAL]],
        beta: T,
        alpha: T,
        x_vec: &[[T; NDIMVAL]],
    ) where
        T: num_traits::Float,
    {
        use del_geo_core::matn_col_major;
        use del_geo_core::vecn::VecN;
        assert_eq!(y_vec.len(), self.num_blk);
        for m in y_vec.iter_mut() {
            del_geo_core::vecn::scale_in_place(m, beta);
        }
        for i_blk in 0..self.num_blk {
            for idx in self.row2idx[i_blk]..self.row2idx[i_blk + 1] {
                assert!(idx < self.idx2col.len());
                let j_blk = self.idx2col[idx];
                assert!(j_blk < self.num_blk);
                let a = matn_col_major::mult_vec(&self.idx2val[idx], &x_vec[j_blk]).scale(alpha);
                del_geo_core::vecn::add_in_place(&mut y_vec[i_blk], &a);
            }
            {
                let a = matn_col_major::mult_vec(&self.row2val[i_blk], &x_vec[i_blk]).scale(alpha);
                del_geo_core::vecn::add_in_place(&mut y_vec[i_blk], &a);
            }
        }
    }

    /// set zero to all the values
    pub fn set_zero(&mut self) {
        assert_eq!(self.idx2val.len(), self.idx2col.len());
        self.row2val.fill([T::zero(); BLKSIZE]);
        self.idx2val.fill([T::zero(); BLKSIZE]);
    }

    pub fn merge<const NNODE: usize>(
        &mut self,
        emat: &[[[T; BLKSIZE]; NNODE]; NNODE],
        node2vtx: &[usize; NNODE],
        col2idx: &mut Vec<usize>,
    ) {
        col2idx.resize(self.num_blk, usize::MAX);
        for i_node in 0..NNODE {
            let i_vtx = node2vtx[i_node];
            for idx in self.row2idx[i_vtx]..self.row2idx[i_vtx + 1] {
                let j_vtx = self.idx2col[idx];
                col2idx[j_vtx] = idx;
            }
            for j_node in 0..NNODE {
                if i_node == j_node {
                    del_geo_core::matn_col_major::add_in_place(
                        &mut self.row2val[i_vtx],
                        &emat[i_node][j_node],
                    );
                } else {
                    let j_vtx = node2vtx[j_node];
                    let idx0 = col2idx[j_vtx];
                    assert_ne!(idx0, usize::MAX);
                    del_geo_core::matn_col_major::add_in_place(
                        &mut self.idx2val[idx0],
                        &emat[i_node][j_node],
                    );
                }
            }
            for idx in self.row2idx[i_vtx]..self.row2idx[i_vtx + 1] {
                let j_vtx = self.idx2col[idx];
                col2idx[j_vtx] = usize::MAX;
            }
        }
    }
}

pub struct MatrixRefMut<'a, T, const BLKSIZE: usize> {
    pub num_blk: usize,
    pub row2idx: &'a [usize],
    pub idx2col: &'a [usize],
    pub idx2val: &'a mut [[T; BLKSIZE]],
    pub row2val: &'a mut [[T; BLKSIZE]],
}

impl<'a, T, const BLKSIZE: usize> MatrixRefMut<'a, T, BLKSIZE>
where
    T: num_traits::Float,
{
    pub fn set_zero(&mut self) {
        assert_eq!(self.idx2val.len(), self.idx2col.len());
        self.row2val.fill([T::zero(); BLKSIZE]);
        self.idx2val.fill([T::zero(); BLKSIZE]);
    }

    pub fn merge_for_array_blk<const NNODE: usize>(
        &mut self,
        emat: &[[[T; BLKSIZE]; NNODE]; NNODE],
        node2vtx: &[usize; NNODE],
        col2idx: &mut Vec<usize>,
    ) {
        col2idx.resize(self.num_blk, usize::MAX);
        for i_node in 0..NNODE {
            let i_vtx = node2vtx[i_node];
            for idx in self.row2idx[i_vtx]..self.row2idx[i_vtx + 1] {
                let j_vtx = self.idx2col[idx];
                col2idx[j_vtx] = idx;
            }
            for j_node in 0..NNODE {
                if i_node == j_node {
                    del_geo_core::matn_col_major::add_in_place(
                        &mut self.row2val[i_vtx],
                        &emat[i_node][j_node],
                    );
                } else {
                    let j_vtx = node2vtx[j_node];
                    let idx0 = col2idx[j_vtx];
                    assert_ne!(idx0, usize::MAX);
                    del_geo_core::matn_col_major::add_in_place(
                        &mut self.idx2val[idx0],
                        &emat[i_node][j_node],
                    );
                }
            }
            for idx in self.row2idx[i_vtx]..self.row2idx[i_vtx + 1] {
                let j_vtx = self.idx2col[idx];
                col2idx[j_vtx] = usize::MAX;
            }
        }
    }
}

pub struct MatrixRef<'a, T, const BLKSIZE: usize> {
    pub num_blk: usize,
    pub row2idx: &'a [usize],
    pub idx2col: &'a [usize],
    pub idx2val: &'a [[T; BLKSIZE]],
    pub row2val: &'a [[T; BLKSIZE]],
}

impl<'a, T, const BLKSIZE: usize> MatrixRef<'a, T, BLKSIZE>
where
    T: num_traits::Float,
{
    /// generalized matrix-vector multiplication
    /// where matrix is sparse (not block) matrix
    /// `{y_vec} <- \alpha * [a_mat] * {x_vec} + \beta * {y_vec}`
    pub fn mult_vec<const NDIMVAL: usize>(
        &self,
        y_vec: &mut [[T; NDIMVAL]],
        beta: T,
        alpha: T,
        x_vec: &[[T; NDIMVAL]],
    ) where
        T: num_traits::Float,
    {
        use del_geo_core::matn_col_major;
        use del_geo_core::vecn::VecN;
        assert_eq!(y_vec.len(), self.num_blk);
        for m in y_vec.iter_mut() {
            del_geo_core::vecn::scale_in_place(m, beta);
        }
        for i_blk in 0..self.num_blk {
            for idx in self.row2idx[i_blk]..self.row2idx[i_blk + 1] {
                assert!(idx < self.idx2col.len());
                let j_blk = self.idx2col[idx];
                assert!(j_blk < self.num_blk);
                let a = matn_col_major::mult_vec(&self.idx2val[idx], &x_vec[j_blk]).scale(alpha);
                del_geo_core::vecn::add_in_place(&mut y_vec[i_blk], &a);
            }
            {
                let a = matn_col_major::mult_vec(&self.row2val[i_blk], &x_vec[i_blk]).scale(alpha);
                del_geo_core::vecn::add_in_place(&mut y_vec[i_blk], &a);
            }
        }
    }
}
