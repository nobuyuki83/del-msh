#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Tensor};

#[allow(dead_code)]
fn compute_residual_norm(
    idx2vtx: &[u32],
    vtx2idx: &[u32],
    vtx2trgs: &[f32],
    vtx2vars: &[f32],
    lambda: f32,
) -> f32 {
    let num_vtx = vtx2trgs.len() / 3;
    let func_res = |i_vtx: usize| -> f32 {
        let mut res = [
            vtx2trgs[i_vtx * 3],
            vtx2trgs[i_vtx * 3 + 1],
            vtx2trgs[i_vtx * 3 + 2],
        ];
        for &j_vtx in &idx2vtx[vtx2idx[i_vtx] as usize..vtx2idx[i_vtx + 1] as usize] {
            let j_vtx = j_vtx as usize;
            res[0] += lambda * vtx2vars[j_vtx * 3];
            res[1] += lambda * vtx2vars[j_vtx * 3 + 1];
            res[2] += lambda * vtx2vars[j_vtx * 3 + 2];
        }
        let valence = (vtx2idx[i_vtx + 1] - vtx2idx[i_vtx]) as f32;
        res[0] -= (1f32 + lambda * valence) * vtx2vars[i_vtx * 3];
        res[1] -= (1f32 + lambda * valence) * vtx2vars[i_vtx * 3 + 1];
        res[2] -= (1f32 + lambda * valence) * vtx2vars[i_vtx * 3 + 2];
        res[0] * res[0] + res[1] * res[1] + res[2] * res[2]
    };
    (0..num_vtx).map(func_res).sum()
}

/// minimizer for f(`vtx2vals`) = || `vtx2vals` - `vtx2trgs`||^2 + `lambda` * tr(`vtx2vals`^T * L *`vtx2vals`)
/// where L is the graph Laplacian
///
/// grad f(`vtx2vals`) = `vtx2vals` - `vtx2trgs` + `lambda` * L * `vtx2vals` = 0
/// => (I + `lambda` * L)`vtx2vals` = `vtx2trgs`
pub struct LaplacianSmoothing {
    pub lambda: f32,
    pub vtx2idx: Tensor,
    pub idx2vtx: Tensor,
    pub num_iter: usize,
}

impl CustomOp1 for LaplacianSmoothing {
    fn name(&self) -> &'static str {
        "laplacian smoothing"
    }
    fn cpu_fwd(
        &self,
        vtx2trgs: &CpuStorage,
        l_vtx2trgs: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let num_vtx = l_vtx2trgs.dim(0)?;
        let num_dim = l_vtx2trgs.dim(1)?;
        let num_dof = num_vtx * num_dim;
        use std::ops::Deref;
        let vtx2trgs = vtx2trgs.as_slice::<f32>()?;
        get_cpu_slice_and_storage_from_tensor!(vtx2idx, storage_vtx2idx, self.vtx2idx, u32);
        get_cpu_slice_and_storage_from_tensor!(idx2vtx, storage_idx2vtx, self.idx2vtx, u32);
        let func_upd = |i_vtx: usize, lhs_next: &mut [f32], vtx2lhs_prev: &[f32]| {
            let mut rhs = [
                vtx2trgs[i_vtx * 3],
                vtx2trgs[i_vtx * 3 + 1],
                vtx2trgs[i_vtx * 3 + 2],
            ];
            for &j_vtx in &idx2vtx[vtx2idx[i_vtx] as usize..vtx2idx[i_vtx + 1] as usize] {
                let j_vtx = j_vtx as usize;
                rhs[0] += self.lambda * vtx2lhs_prev[j_vtx * 3];
                rhs[1] += self.lambda * vtx2lhs_prev[j_vtx * 3 + 1];
                rhs[2] += self.lambda * vtx2lhs_prev[j_vtx * 3 + 2];
            }
            let valence = (vtx2idx[i_vtx + 1] - vtx2idx[i_vtx]) as f32;
            let inv_dia = 1f32 / (1f32 + self.lambda * valence);
            lhs_next[0] = rhs[0] * inv_dia;
            lhs_next[1] = rhs[1] * inv_dia;
            lhs_next[2] = rhs[2] * inv_dia;
        };
        //let mut conv_hist: Vec<f32> = vec![]; // this will be returned
        let mut vtx2vars = vtx2trgs.to_vec(); // this will be returned
        let mut vtx2vars_tmp = vec![0f32; num_dof]; // this is a temp buffer
        use rayon::prelude::*;
        for _iter in 0..self.num_iter {
            vtx2vars_tmp
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_vtx, lhs1)| func_upd(i_vtx, lhs1, &vtx2vars));
            vtx2vars
                .par_chunks_mut(3)
                .enumerate()
                .for_each(|(i_vtx, lhs)| func_upd(i_vtx, lhs, &vtx2vars_tmp));
            //let res = compute_residual_norm(idx2vtx, vtx2idx, vtx2trgs, &vtx2vars, self.lambda);
            //conv_hist.push(res);
        }
        Ok((CpuStorage::F32(vtx2vars), (num_vtx, num_dim).into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        vtx2trgs: &CudaStorage,
        l_vtx2trgs: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::WrapErr;
        let device = vtx2trgs.device();
        let num_vtx = l_vtx2trgs.dim(0)?;
        let num_dim = l_vtx2trgs.dim(1)?;
        let num_dof = num_vtx * num_dim;
        use std::ops::Deref;
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            vtx2idx,
            s_vtx2idx,
            l_vtx2idx,
            self.vtx2idx,
            u32
        );
        assert_eq!(l_vtx2idx.dims(), &[num_vtx + 1]);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            idx2vtx,
            s_idx2vtx,
            _l_idx2vtx,
            self.idx2vtx,
            u32
        );
        let vtx2trgs = vtx2trgs.as_cuda_slice::<f32>()?;
        let mut vtx2vars = unsafe { device.alloc::<f32>(num_dof)? }; // this will be returned
        device.memcpy_dtod(vtx2trgs, &mut vtx2vars)?;
        let mut vtx2vars_tmp = unsafe { device.alloc::<f32>(num_dof)? }; // this is a temp buffer
        for _iter in 0..self.num_iter {
            del_msh_cudarc_safe::laplacian_smoothing_jacobi::solve(
                &device.cuda_stream(),
                vtx2idx,
                idx2vtx,
                self.lambda,
                &mut vtx2vars_tmp.slice_mut(..),
                &vtx2vars,
                vtx2trgs,
            )
            .w()?;
            del_msh_cudarc_safe::laplacian_smoothing_jacobi::solve(
                &device.cuda_stream(),
                vtx2idx,
                idx2vtx,
                self.lambda,
                &mut vtx2vars.slice_mut(..),
                &vtx2vars_tmp,
                vtx2trgs,
            )
            .w()?;
        }
        let cuda_storage = CudaStorage::wrap_cuda_slice(vtx2vars, device.clone());
        Ok((cuda_storage, (num_vtx, num_dim).into()))
    }
}

#[test]
#[allow(unused_variables)]
fn test_laplacian_smoothing() -> candle_core::Result<()> {
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh3_primitive::sphere_yup(1f32, 512, 512);
    let num_vtx = vtx2xyz.len() / 3;
    println!("num_vtx: {num_vtx}");
    let (vtx2idx, idx2vtx) =
        del_msh_cpu::vtx2vtx::from_uniform_mesh::<u32>(&tri2vtx, 3, num_vtx, false);
    let n = vtx2idx.len();
    let vtx2idx = Tensor::from_vec(vtx2idx, n, &candle_core::Device::Cpu)?;
    let n = idx2vtx.len();
    let idx2vtx = Tensor::from_vec(idx2vtx, n, &candle_core::Device::Cpu)?;
    let vtx2vars_in = Tensor::rand(-1f32, 1f32, (num_vtx, 3), &candle_core::Device::Cpu)?;
    let layer = LaplacianSmoothing {
        lambda: 1.0,
        num_iter: 100,
        vtx2idx: vtx2idx.clone(),
        idx2vtx: idx2vtx.clone(),
    };
    let vtx2out = &vtx2vars_in.apply_op1_no_bwd(&layer)?;
    let vtx2out_cpu = vtx2out.flatten_all()?.to_vec1::<f32>()?;
    #[cfg(feature = "cuda")]
    {
        let device = candle_core::Device::cuda_if_available(0)?;
        let vtx2idx = vtx2idx.to_device(&device)?;
        let idx2vtx = idx2vtx.to_device(&device)?;
        let vtx2vars_in = vtx2vars_in.to_device(&device)?;
        let layer = LaplacianSmoothing {
            lambda: 1.0,
            num_iter: 100,
            vtx2idx: vtx2idx.clone(),
            idx2vtx: idx2vtx.clone(),
        };
        let vtx2out = vtx2vars_in.apply_op1_no_bwd(&layer)?;
        let vtx2out_cuda = vtx2out.flatten_all()?.to_vec1::<f32>()?;
        vtx2out_cpu
            .iter()
            .zip(vtx2out_cuda.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    }
    Ok(())
}
