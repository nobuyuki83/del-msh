use std::ops::Deref;

use candle_core::{CpuStorage, Layout, Shape, Tensor};

pub struct Layer {}

impl candle_core::CustomOp1 for crate::diffcoord_polyloop2::Layer {
    fn name(&self) -> &'static str {
        "polyloop to diffcoord"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (num_vtx, two) = layout.shape().dims2()?;
        assert_eq!(two, 2);
        let vtx2xy = storage.as_slice::<f32>()?;
        let mut vtx2diff = vec![0f32; num_vtx * 2];
        for i_edge in 0..num_vtx {
            let i0_vtx = (i_edge + num_vtx - 1) % num_vtx;
            let i1_vtx = i_edge;
            let i2_vtx = (i_edge + 1) % num_vtx;
            for i_dim in 0..2 {
                vtx2diff[i1_vtx * 2 + i_dim] -= 0.5f32 * vtx2xy[i0_vtx * 2 + i_dim];
                vtx2diff[i1_vtx * 2 + i_dim] += 1.0f32 * vtx2xy[i1_vtx * 2 + i_dim];
                vtx2diff[i1_vtx * 2 + i_dim] -= 0.5f32 * vtx2xy[i2_vtx * 2 + i_dim];
            }
        }
        let shape = candle_core::Shape::from((num_vtx, 2));
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
        let (num_vtx, two) = dw_vtx2diff.shape().dims2()?;
        assert_eq!(two, 2);
        let dw_vtx2diff = dw_vtx2diff.storage_and_layout().0;
        let dw_vtx2diff = match dw_vtx2diff.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<f32>()?,
            _ => panic!(),
        };
        let mut dw_vtx2xy = vec![0f32; num_vtx * 2];
        for i_edge in 0..num_vtx {
            let i0_vtx = (i_edge + num_vtx - 1) % num_vtx;
            let i1_vtx = i_edge;
            let i2_vtx = (i_edge + 1) % num_vtx;
            for i_dim in 0..2 {
                dw_vtx2xy[i0_vtx * 2 + i_dim] -= 0.5 * dw_vtx2diff[i1_vtx * 2 + i_dim];
                dw_vtx2xy[i1_vtx * 2 + i_dim] += 1.0 * dw_vtx2diff[i1_vtx * 2 + i_dim];
                dw_vtx2xy[i2_vtx * 2 + i_dim] -= 0.5 * dw_vtx2diff[i1_vtx * 2 + i_dim];
            }
        }
        let dw_vtx2xy = Tensor::from_vec(
            dw_vtx2xy,
            candle_core::Shape::from((num_vtx, 2)),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_vtx2xy))
    }
}

#[test]
fn edge_length_constraint() -> anyhow::Result<()> {
    let num_vtx = 16;
    // let edge_length = 2.0f32 * std::f32::consts::PI / num_vtx as f32;
    let mut vtx2xy = del_msh_core::polyloop2::from_circle(1.0, num_vtx);
    {
        use rand::Rng;
        let mut rng = rand::rng();
        for vtx in vtx2xy.chunks_mut(2) {
            vtx[0] += rng.random::<f32>();
            vtx[1] += rng.random::<f32>();
        }
    }
    let vtx2xy = {
        candle_core::Var::from_slice(
            vtx2xy.as_slice(),
            candle_core::Shape::from((vtx2xy.len() / 2, 2)),
            &candle_core::Device::Cpu,
        )
        .unwrap()
    };
    for iter in 0..200 {
        let render = Layer {};
        let vtx2diff = vtx2xy.apply_op1(render)?;
        assert_eq!(vtx2diff.shape(), vtx2xy.shape());
        let loss_straight = vtx2diff.sqr()?.sum_all()?;
        {
            let val_loss = loss_straight.to_vec0::<f32>()?;
            dbg!(val_loss);
        }
        let grad = loss_straight.backward()?;
        let dw_vtx2xyz = grad.get(&vtx2xy).unwrap();
        let _ = vtx2xy.set(&vtx2xy.as_tensor().sub(&(dw_vtx2xyz * 0.1)?)?);
        if iter % 10 == 0 {
            let vtx2xy: Vec<_> = vtx2xy.flatten_all()?.to_vec1::<f32>()?;
            del_msh_core::io_obj::save_vtx2xyz_as_polyloop(
                format!("../target/polyloop_{}.obj", iter),
                &vtx2xy,
                2,
            )?;
        }
    }
    Ok(())
}
