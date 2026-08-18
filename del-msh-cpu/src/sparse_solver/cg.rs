use crate::sparse_solver::sparse_square::{MatrixOwned, MatrixRef};

/// solve linear system using the Conjugate Gradient (CG) method
pub fn conjugate_gradient<T, const NDIMVAL: usize, const BLKSIZE: usize>(
    r_vec: &mut [[T; NDIMVAL]],
    u_vec: &mut [[T; NDIMVAL]],
    ap_vec: &mut [[T; NDIMVAL]],
    p_vec: &mut [[T; NDIMVAL]],
    conv_ratio_tol: T,
    max_iteration: usize,
    mat: MatrixRef<T, BLKSIZE>,
) -> Vec<T>
where
    T: num_traits::Float + std::fmt::Display + std::fmt::Debug,
{
    use crate::sparse_solver::*;
    let _num_dim = r_vec.len() / mat.row2val.len();
    //
    let mut conv_hist = Vec::<T>::new();
    slice_of_array::set_zero(u_vec);
    let mut sqnorm_res = slice_of_array::dot(r_vec, r_vec);
    if sqnorm_res < T::epsilon() {
        return conv_hist;
    }
    let inv_sqnorm_res_ini = T::one() / sqnorm_res;
    slice_of_array::copy(p_vec, r_vec); // {p} = {r}  (set initial serch direction, copy value not reference)
    for _iitr in 0..max_iteration {
        // alpha = (r,r) / (p,Ap)
        mat.mult_vec::<NDIMVAL>(ap_vec, T::zero(), T::one(), p_vec); // {Ap_vec} = [mat]*{p_vec}
        let pap = slice_of_array::dot(p_vec, ap_vec);
        // assert!(pap >= T::zero(), "{pap}");
        let alpha = sqnorm_res / pap;
        slice_of_array::add_scaled_vector(u_vec, alpha, p_vec); // {u} = +alpha*{p} + {u} (update x)
        slice_of_array::add_scaled_vector(r_vec, -alpha, ap_vec); // {r} = -alpha*{Ap} + {r}
        let sqnorm_res_new = slice_of_array::dot(r_vec, r_vec);
        let conv_ratio = (sqnorm_res_new * inv_sqnorm_res_ini).sqrt();
        conv_hist.push(conv_ratio);
        if conv_ratio < conv_ratio_tol {
            return conv_hist;
        }
        {
            let beta = sqnorm_res_new / sqnorm_res; // beta = (r1,r1) / (r0,r0)
            sqnorm_res = sqnorm_res_new;
            slice_of_array::scale_and_add_vec(p_vec, beta, r_vec); // {p} = {r} + beta*{p}
        }
    }
    conv_hist
}

/// solve a real-valued linear system using the conjugate gradient method with preconditioner
#[allow(clippy::too_many_arguments)]
pub fn preconditioned_conjugate_gradient<T, const NDIMVAL: usize, const BLKSIZE: usize>(
    r_vec: &mut [[T; NDIMVAL]],
    x_vec: &mut Vec<[T; NDIMVAL]>,
    pr_vec: &mut Vec<[T; NDIMVAL]>,
    p_vec: &mut Vec<[T; NDIMVAL]>,
    conv_ratio_tol: T,
    max_nitr: usize,
    mat: &MatrixOwned<[T; BLKSIZE]>,
    ilu: &crate::sparse_solver::sparse_ilu::Preconditioner<[T; BLKSIZE]>,
) -> Vec<T>
where
    T: num_traits::Float + std::fmt::Debug,
{
    use crate::sparse_solver::slice_of_array::{
        add_scaled_vector, copy, dot, scale_and_add_vec, set_zero,
    };
    {
        let n = r_vec.len();
        x_vec.resize(n, [T::zero(); NDIMVAL]);
        pr_vec.resize(n, [T::zero(); NDIMVAL]);
        p_vec.resize(n, [T::zero(); NDIMVAL]);
    }
    assert_eq!(r_vec.len(), mat.num_blk);
    let mut conv_hist = Vec::<T>::new();

    set_zero(x_vec);

    let inv_sqnorm_res0 = {
        let sqnorm_res0 = dot(r_vec, r_vec); // DotX(r_vec, r_vec, N);
        conv_hist.push(sqnorm_res0.sqrt());
        if sqnorm_res0 < T::epsilon() {
            return conv_hist;
        }
        T::one() / sqnorm_res0
    };

    // {Pr} = [P]{r}
    copy(pr_vec, r_vec); // std::vector<double> Pr_vec(r_vec, r_vec + N);

    crate::sparse_solver::sparse_ilu::solve_preconditioning_vec(pr_vec, ilu); // ilu.SolvePrecond(Pr_vec.data());

    // {p} = {Pr}
    copy(p_vec, pr_vec);

    // rPr = ({r},{Pr})
    let mut rpr = dot(r_vec, pr_vec); // DotX(r_vec, Pr_vec.data(), N);
    for _iitr in 0..max_nitr {
        // {Ap} = [A]{p}
        mat.mult_vec(pr_vec, T::zero(), T::one(), p_vec);
        {
            // alpha = ({r},{Pr})/({p},{Ap})
            let pap = dot(p_vec, pr_vec);
            let alpha = rpr / pap;
            add_scaled_vector(r_vec, -alpha, pr_vec); // {r} = -alpha*{Ap} + {r}
            add_scaled_vector(x_vec, alpha, p_vec); // {x} = +alpha*{p} + {x}
        }
        {
            // Converge Judgement
            let sqnorm_res = dot(r_vec, r_vec); // DotX(r_vec, r_vec, N);
            conv_hist.push(sqnorm_res.sqrt());
            let conv_ratio = (sqnorm_res * inv_sqnorm_res0).sqrt();
            if conv_ratio < conv_ratio_tol {
                return conv_hist;
            }
        }
        {
            // calc beta
            copy(pr_vec, r_vec);
            // {Pr} = [P]{r}
            crate::sparse_solver::sparse_ilu::solve_preconditioning_vec(pr_vec, ilu);
            // rPr1 = ({r},{Pr})
            let rpr1 = dot(r_vec, pr_vec);
            // beta = rPr1/rPr
            let beta = rpr1 / rpr;
            rpr = rpr1;
            // {p} = {Pr} + beta*{p}
            scale_and_add_vec(p_vec, beta, pr_vec);
        }
    }
    {
        // Converge Judgement
        let sq_norm_res = dot(r_vec, r_vec); // DotX(r_vec, r_vec, N);
        conv_hist.push(sq_norm_res.sqrt());
    }
    conv_hist
}
