use std::ops::Deref;

use candle_core::Device::Cpu;
use candle_core::{CpuStorage, Layout, Shape, Tensor};

pub struct Layer {
    pub vtx2idx: Tensor,
    pub idx2vtx: Tensor,
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "diffcoord for trimesh3"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (num_vtx, three) = layout.shape().dims2()?;
        assert_eq!(three, 3);
        let vtx2xyz = storage.as_slice::<f32>()?;
        let vtx2idx = self.vtx2idx.storage_and_layout().0;
        let vtx2idx = match vtx2idx.deref() {
            candle_core::Storage::Cpu(cpu_vtx2idx) => cpu_vtx2idx.as_slice::<u32>()?,
            _ => panic!(),
        };
        let idx2vtx = self.idx2vtx.storage_and_layout().0;
        let idx2vtx = match idx2vtx.deref() {
            candle_core::Storage::Cpu(cpu_idx2vtx) => cpu_idx2vtx.as_slice::<u32>()?,
            _ => panic!(),
        };
        //
        let mut vtx2diff = Vec::<f32>::from(vtx2xyz);
        for i_vtx in 0..num_vtx {
            let velence_inv = 1f32 / (vtx2idx[i_vtx + 1] - vtx2idx[i_vtx]) as f32;
            let idx0 = vtx2idx[i_vtx] as usize;
            let idx1 = vtx2idx[i_vtx + 1] as usize;
            for &j_vtx in &idx2vtx[idx0..idx1] {
                let j_vtx = j_vtx as usize;
                for i_dim in 0..3 {
                    vtx2diff[i_vtx * 3 + i_dim] -= velence_inv * vtx2xyz[j_vtx * 3 + i_dim];
                }
            }
        }
        let shape = candle_core::Shape::from((num_vtx, 3));
        let storage = candle_core::WithDType::to_cpu_storage_owned(vtx2diff);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(
        &self,
        _vtx2xy: &Tensor,
        _vtx2diff: &Tensor,
        dw_vtx2diff: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (num_vtx, three) = dw_vtx2diff.shape().dims2()?;
        assert_eq!(three, 3);
        let vtx2idx = self.vtx2idx.storage_and_layout().0;
        let vtx2idx = match vtx2idx.deref() {
            candle_core::Storage::Cpu(cpu_vtx2idx) => cpu_vtx2idx.as_slice::<u32>()?,
            _ => panic!(),
        };
        let idx2vtx = self.idx2vtx.storage_and_layout().0;
        let idx2vtx = match idx2vtx.deref() {
            candle_core::Storage::Cpu(cpu_idx2vtx) => cpu_idx2vtx.as_slice::<u32>()?,
            _ => panic!(),
        };
        let dw_vtx2diff = dw_vtx2diff.storage_and_layout().0;
        let dw_vtx2diff = match dw_vtx2diff.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<f32>()?,
            _ => panic!(),
        };
        let mut dw_vtx2xyz = vec![0f32; num_vtx * 3];
        for i_vtx in 0..num_vtx {
            let velence_inv = 1f32 / (vtx2idx[i_vtx + 1] - vtx2idx[i_vtx]) as f32;
            let idx0 = vtx2idx[i_vtx] as usize;
            let idx1 = vtx2idx[i_vtx + 1] as usize;
            for &j_vtx in &idx2vtx[idx0..idx1] {
                let j_vtx = j_vtx as usize;
                dw_vtx2xyz[j_vtx * 3] -= velence_inv * dw_vtx2diff[i_vtx * 3];
                dw_vtx2xyz[j_vtx * 3 + 1] -= velence_inv * dw_vtx2diff[i_vtx * 3 + 1];
                dw_vtx2xyz[j_vtx * 3 + 2] -= velence_inv * dw_vtx2diff[i_vtx * 3 + 2];
            }
            dw_vtx2xyz[i_vtx * 3] += dw_vtx2diff[i_vtx * 3];
            dw_vtx2xyz[i_vtx * 3 + 1] += dw_vtx2diff[i_vtx * 3 + 1];
            dw_vtx2xyz[i_vtx * 3 + 2] += dw_vtx2diff[i_vtx * 3 + 2];
        }
        let dw_vtx2xyz =
            Tensor::from_vec(dw_vtx2xyz, candle_core::Shape::from((num_vtx, 3)), &Cpu)?;
        Ok(Some(dw_vtx2xyz))
    }
}

#[test]
fn test_backward() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz0) = del_msh_cpu::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 32, 32);
    let vtx2vtx =
        del_msh_cpu::vtx2vtx::from_uniform_mesh::<u32>(&tri2vtx, 3, vtx2xyz0.len() / 3, false);

    let num_vtx = vtx2xyz0.len() / 3;
    let vtx2xyz0 = candle_core::Var::from_vec(vtx2xyz0, (num_vtx, 3), &Cpu)?;

    let num_vtx2idx = vtx2vtx.0.len();
    let vtx2idx = Tensor::from_vec(vtx2vtx.0, num_vtx2idx, &Cpu)?;

    let num_idx2vtx = vtx2vtx.1.len();
    let idx2vtx = Tensor::from_vec(vtx2vtx.1, num_idx2vtx, &Cpu)?;

    let trg = Tensor::rand_like(&vtx2xyz0, -1., 1.)?;

    let (loss0, dw_vtx2xyz) = {
        let render = Layer {
            vtx2idx: vtx2idx.clone(),
            idx2vtx: idx2vtx.clone(),
        };
        let vtx2diff0 = vtx2xyz0.apply_op1(render)?;
        let loss0 = (vtx2diff0.mul(&trg))?.sum_all()?;
        let loss0_val = loss0.to_vec0::<f32>()?;
        let gradtore = loss0.backward()?;
        let grad = gradtore.get(&vtx2xyz0).unwrap().to_owned();
        let grad = grad.flatten_all()?.to_vec1::<f32>()?;
        (loss0_val, grad)
    };
    let eps: f32 = 1.0e-3;
    for i_vtx in 0..num_vtx {
        for i_dim in 0..3 {
            let vtx2xyz1 =
                crate::perturb_tensor::peturb_2d_tensor(&vtx2xyz0, i_vtx, i_dim, eps as f64)?;
            let render = Layer {
                vtx2idx: vtx2idx.clone(),
                idx2vtx: idx2vtx.clone(),
            };
            let vtx2diff1 = vtx2xyz1.apply_op1(render)?;
            let loss1 = (vtx2diff1.mul(&trg))?.sum_all()?;
            let loss1 = loss1.to_vec0::<f32>()?;
            let val_num = (loss1 - loss0) / eps;
            let val_ana = dw_vtx2xyz[i_vtx * 3 + i_dim];
            assert!(
                (val_num - val_ana).abs() < 0.01 * (val_ana.abs() + 0.1),
                "{} {}",
                val_ana,
                val_num
            );
        }
    }
    Ok(())
}
