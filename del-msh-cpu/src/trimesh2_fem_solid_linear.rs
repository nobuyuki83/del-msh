pub fn add_weighted_emat_2d<T, const NNO: usize>(
    emat: &mut [[[T; 4]; NNO]; NNO],
    lambda: T,
    myu: T,
    dldx: [[T; NNO]; 2],
    w: T,
) where
    T: num_traits::Float + std::ops::AddAssign,
{
    for (ino, jno) in itertools::iproduct!(0..NNO, 0..NNO) {
        emat[ino][jno][0] += w * (lambda + myu) * dldx[0][ino] * dldx[0][jno];
        emat[ino][jno][2] +=
            w * (lambda * dldx[0][ino] * dldx[1][jno] + myu * dldx[0][jno] * dldx[1][ino]);
        emat[ino][jno][1] +=
            w * (lambda * dldx[1][ino] * dldx[0][jno] + myu * dldx[1][jno] * dldx[0][ino]);
        emat[ino][jno][3] += w * (lambda + myu) * dldx[1][ino] * dldx[1][jno];
        let dtmp1 = w * myu * (dldx[1][ino] * dldx[1][jno] + dldx[0][ino] * dldx[0][jno]);
        emat[ino][jno][0] += dtmp1;
        emat[ino][jno][3] += dtmp1;
    }
}

pub fn emat_tri2<T>(lambda: T, myu: T, p0: &[T; 2], p1: &[T; 2], p2: &[T; 2]) -> [[[T; 4]; 3]; 3]
where
    T: num_traits::Float + 'static + Copy + std::ops::AddAssign,
    f64: num_traits::AsPrimitive<T>,
{
    let area = del_geo_core::tri2::area(p0, p1, p2);
    let (dldx, _) = del_geo_core::tri2::dldx(p0, p1, p2);
    let mut emat = [[[T::zero(); 4]; 3]; 3];
    add_weighted_emat_2d::<T, 3>(&mut emat, lambda, myu, dldx, area);
    emat
}

pub fn mult_emat_evec(emat: &[[[f64; 4]; 3]; 3], avec: &[[f64; 2]; 3]) -> [[f64; 2]; 3] {
    use del_geo_core::mat2_col_major::Mat2ColMajor;
    let mut r = [[0f64; 2]; 3];
    for ino in 0..3 {
        for jno in 0..3 {
            let a = emat[ino][jno].mult_vec(&avec[jno]);
            del_geo_core::vec2::add_in_place(&mut r[ino], &a);
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy(
        tri2vtx: &[[usize; 3]],
        vtx2xy_ini: &[[f64; 2]],
        vtx2xy_def: &[[f64; 2]],
        lambda: f64,
        myu: f64,
    ) -> f64 {
        let mut eng = 0.0;
        for i_tri in 0..tri2vtx.len() {
            let node2vtx = &tri2vtx[i_tri];
            let p0 = &vtx2xy_ini[node2vtx[0]];
            let p1 = &vtx2xy_ini[node2vtx[1]];
            let p2 = &vtx2xy_ini[node2vtx[2]];
            let emat = emat_tri2(lambda, myu, p0, p1, p2);
            {
                use del_geo_core::vec2::Vec2;
                let q0 = &vtx2xy_def[node2vtx[0]];
                let q1 = &vtx2xy_def[node2vtx[1]];
                let q2 = &vtx2xy_def[node2vtx[2]];
                let u = [q0.sub(&p0), q1.sub(&p1), q2.sub(&p2)];
                let er = mult_emat_evec(&emat, &u);
                let e_eng = er[0].dot(&u[0]) + er[1].dot(&u[1]) + er[2].dot(&u[2]);
                let e_eng = e_eng * 0.5;
                assert!(e_eng >= -1.0e-10, "{e_eng}");
                eng += e_eng;
            }
        }
        eng
    }

    struct Problem {
        tri2vtx: Vec<[usize; 3]>,
        vtx2xy_ini: Vec<[f64; 2]>,
        vtx2xy_def: Vec<[f64; 2]>,
        bsm: crate::sparse_solver::sparse_square::MatrixOwned<[f64; 4]>,
        r_vec: Vec<[f64; 2]>,
        lambda: f64,
        myu: f64,
    }

    fn make_problem(lambda: f64, myu: f64) -> Problem {
        let (tri2vtx, vtx2xy_ini) = crate::trimesh2_dynamic::meshing_from_polyloop2::<usize, f64>(
            &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            0.07,
            0.07,
        );
        let num_vtx = vtx2xy_ini.len();
        crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(
            "../target/linear_solid_2d_ini.obj",
            &tri2vtx,
            &vtx2xy_ini
                .iter()
                .map(|xy| [xy[0], xy[1], 0.])
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut bsm = {
            let vtx2vtx = crate::vtx2vtx::from_uniform_mesh(&tri2vtx, num_vtx, false);
            let num_idx = vtx2vtx.1.len();
            let bsm = crate::sparse_solver::sparse_square::MatrixOwned {
                num_blk: num_vtx,
                row2idx: vtx2vtx.0,
                idx2col: vtx2vtx.1,
                idx2val: vec![[0.; 4]; num_idx],
                row2val: vec![[0.; 4]; num_vtx],
            };
            bsm
        };
        let mut vtx2xy_def = vtx2xy_ini.clone();
        //
        use rand::RngExt;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0);
        let mut vtx2flg = vec![[0i32; 2]; num_vtx];
        for i_vtx in 0..num_vtx {
            let y = vtx2xy_ini[i_vtx][1];
            if y < 0.001 {
                vtx2flg[i_vtx][0] = 1;
                vtx2flg[i_vtx][1] = 1;
            } else {
                vtx2xy_def[i_vtx][0] += 0.1 * (2.0 * rng.random::<f64>() - 1.0);
                vtx2xy_def[i_vtx][1] += 0.1 * (2.0 * rng.random::<f64>() - 1.0);
            }
        }
        crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(
            "../target/linear_solid_2d_def0.obj",
            &tri2vtx,
            &vtx2xy_def
                .iter()
                .map(|xy| [xy[0], xy[1], 0.])
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut r_vec = vec![[0f64; 2]; num_vtx];
        bsm.set_zero();
        {
            // merge
            let mut row2idx = vec![usize::MAX; num_vtx];
            for i_tri in 0..tri2vtx.len() {
                let node2vtx = &tri2vtx[i_tri];
                let p0 = &vtx2xy_ini[node2vtx[0]];
                let p1 = &vtx2xy_ini[node2vtx[1]];
                let p2 = &vtx2xy_ini[node2vtx[2]];
                let emat = emat_tri2(lambda, myu, p0, p1, p2);
                {
                    use del_geo_core::vec2::Vec2;
                    let q0 = &vtx2xy_def[node2vtx[0]];
                    let q1 = &vtx2xy_def[node2vtx[1]];
                    let q2 = &vtx2xy_def[node2vtx[2]];
                    let u = [q0.sub(&p0), q1.sub(&p1), q2.sub(&p2)];
                    let er = mult_emat_evec(&emat, &u);
                    r_vec[node2vtx[0]].sub_in_place(&er[0]);
                    r_vec[node2vtx[1]].sub_in_place(&er[1]);
                    r_vec[node2vtx[2]].sub_in_place(&er[2]);
                }
                bsm.merge_for_array_blk(&emat, node2vtx, &mut row2idx);
            }
            {
                // check if energy value can be expressed as U^T A U
                let eng = energy(&tri2vtx, &vtx2xy_ini, &vtx2xy_def, lambda, myu);
                let mut u = vec![[0f64; 2]; num_vtx];
                for i_vtx in 0..num_vtx {
                    let p = &vtx2xy_ini[i_vtx];
                    let q = &vtx2xy_def[i_vtx];
                    u[i_vtx] = del_geo_core::vec2::sub(&q, &p);
                }
                let mut au = vec![[0f64; 2]; num_vtx];
                bsm.mult_vec::<2>(&mut au, 0.0, 1.0, &u); // {Ap_vec} = [mat]*{p_vec}
                let pap = crate::sparse_solver::slice_of_array::dot(&u, &au) * 0.5;
                assert!((pap - eng).abs() < 1.0e-10);
            }
        }

        // set bc flag
        for i_vtx in 0..num_vtx {
            for i_dof in 0..2 {
                if vtx2flg[i_vtx][i_dof] == 0 {
                    continue;
                }
                r_vec[i_vtx][i_dof] = 0.0;
            }
        }
        bsm.set_fixed_dof::<2>(1.0, &vtx2flg);
        Problem {
            tri2vtx,
            vtx2xy_ini,
            vtx2xy_def,
            bsm,
            r_vec,
            lambda,
            myu,
        }
    }

    enum SolverType {
        CG,
        PcgIlu0,
        PcgIluInf,
        DENSE,
    }

    fn solve_and_check(sol_type: SolverType, prob: &Problem) -> (f64, f64, usize) {
        use crate::sparse_solver::sparse_ilu::{
            copy_value, decompose, solve_preconditioning_vec, Preconditioner,
        };
        let num_vtx = prob.vtx2xy_ini.len();
        let mut r_vec = prob.r_vec.clone();
        // solve linear system
        let (u_vec, num_iter) = match sol_type {
            SolverType::CG => {
                let mut u_vec = vec![[0f64; 2]; num_vtx];
                let mut p_vec = vec![[0f64; 2]; num_vtx];
                let mut ap_vec = vec![[0f64; 2]; num_vtx];
                let hist = crate::sparse_solver::cg::conjugate_gradient(
                    &mut r_vec,
                    &mut u_vec,
                    &mut ap_vec,
                    &mut p_vec,
                    1.0e-5,
                    100,
                    prob.bsm.as_ref(),
                );
                (u_vec, hist.len())
            }
            SolverType::DENSE => {
                let mut prec = Preconditioner::<[f64; 4]>::new();
                prec.initialize_full(num_vtx);
                copy_value(&mut prec, &prob.bsm);
                decompose::<f64, 2, 4>(&mut prec).unwrap();
                let mut u_vec = r_vec.clone();
                solve_preconditioning_vec(&mut u_vec, &prec);
                (u_vec, 1)
            }
            SolverType::PcgIlu0 => {
                let mut prec = Preconditioner::<[f64; 4]>::new();
                prec.initialize_ilu0(&prob.bsm);
                copy_value(&mut prec, &prob.bsm);
                decompose::<f64, 2, 4>(&mut prec).unwrap();
                let mut u_vec = vec![[0f64; 2]; num_vtx];
                let mut pr_vec = vec![[0f64; 2]; num_vtx];
                let mut p_vec = vec![[0f64; 2]; num_vtx];
                let hist = crate::sparse_solver::cg::preconditioned_conjugate_gradient::<f64, 2, 4>(
                    &mut r_vec,
                    &mut u_vec,
                    &mut pr_vec,
                    &mut p_vec,
                    1.0e-5,
                    100,
                    &prob.bsm,
                    &prec,
                );
                (u_vec, hist.len())
            }
            SolverType::PcgIluInf => {
                let mut prec = crate::sparse_solver::sparse_ilu::Preconditioner::<[f64; 4]>::new();
                prec.initialize_iluk(&prob.bsm, usize::MAX);
                copy_value(&mut prec, &prob.bsm);
                decompose::<f64, 2, 4>(&mut prec).unwrap();
                let mut u_vec = vec![[0f64; 2]; num_vtx];
                let mut pr_vec = vec![[0f64; 2]; num_vtx];
                let mut p_vec = vec![[0f64; 2]; num_vtx];
                let hist = crate::sparse_solver::cg::preconditioned_conjugate_gradient::<f64, 2, 4>(
                    &mut r_vec,
                    &mut u_vec,
                    &mut pr_vec,
                    &mut p_vec,
                    1.0e-5,
                    100,
                    &prob.bsm,
                    &prec,
                );
                (u_vec, hist.len())
            }
        };

        let res_dff = {
            use del_geo_core::vec2::Vec2;
            let mut r1_vec = vec![[0f64; 2]; num_vtx];
            prob.bsm.mult_vec::<2>(&mut r1_vec, 0.0, 1.0, &u_vec);
            prob.r_vec
                .iter()
                .zip(r1_vec)
                .map(|v| (v.0).sub(&v.1).norm())
                .reduce(|a, b| a.max(b))
                .unwrap()
        };
        let mut vtx2xy_def1 = prob.vtx2xy_def.clone();
        for i_vtx in 0..num_vtx {
            vtx2xy_def1[i_vtx][0] += 1.0 * u_vec[i_vtx][0];
            vtx2xy_def1[i_vtx][1] += 1.0 * u_vec[i_vtx][1];
        }
        let eng1 = energy(
            &prob.tri2vtx,
            &prob.vtx2xy_ini,
            &vtx2xy_def1,
            prob.lambda,
            prob.myu,
        );
        (res_dff, eng1, num_iter)
    }

    #[test]
    fn test0() {
        let lambda = 1.0;
        let myu = 1.0;
        let prob = make_problem(lambda, myu);
        let (res_diff, eng1, num_iter) = solve_and_check(SolverType::CG, &prob);
        assert!(res_diff < 1.0e-4);
        assert!(eng1 < 1.0e-7);
        assert!(num_iter < 80);
        let (res_diff, eng1, num_iter) = solve_and_check(SolverType::PcgIlu0, &prob);
        assert!(res_diff < 1.0e-4);
        assert!(eng1 < 1.0e-8);
        assert!(num_iter < 36);
        let (res_diff, eng1, num_iter) = solve_and_check(SolverType::PcgIluInf, &prob);
        assert!(res_diff < 1.0e-15);
        assert!(eng1 < 1.0e-29);
        assert_eq!(num_iter, 2);
        let (res_diff, eng1, num_iter) = solve_and_check(SolverType::DENSE, &prob);
        assert!(res_diff < 1.0e-15);
        assert!(eng1 < 1.0e-29);
        assert_eq!(num_iter, 1);
    }
}

//let r0_vec = r_vec.clone();
/*
 */
/*
del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
    "../target/linear_solid_2d_def1.obj",
    &tri2vtx,
    &vtx2xy_def,
    2,
)
    .unwrap();
 */
//
