use std::ops::Deref;

use candle_core::{CpuStorage, Layout, Shape, Tensor};

pub struct Layer {
    pub elem2idx: Vec<usize>,
    pub idx2vtx: Vec<usize>,
}

impl candle_core::CustomOp1 for crate::polygonmesh2_to_cogs::Layer {
    fn name(&self) -> &'static str {
        "polygonmesh2_to_cogs"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        assert_eq!(layout.shape().dims2()?.1, 2);
        let vtx2xy = storage.as_slice::<f32>()?;
        // TODO: from_polygon_mesh_as_faces
        let elem2cog = del_msh_core::elem2center::from_polygon_mesh_as_points(
            &self.elem2idx,
            &self.idx2vtx,
            vtx2xy,
            2,
        );
        let shape = candle_core::Shape::from((elem2cog.len() / 2, 2));
        let storage = candle_core::WithDType::to_cpu_storage_owned(elem2cog);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    fn bwd(
        &self,
        vtx2xy: &Tensor,
        _elem2cog: &Tensor,
        dw_elem2cog: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let dw_elem2cog = dw_elem2cog.storage_and_layout().0;
        let dw_elem2cog = match dw_elem2cog.deref() {
            candle_core::Storage::Cpu(cpu_dw_area) => cpu_dw_area.as_slice::<f32>()?,
            _ => panic!(),
        };
        //
        let (num_vtx, two) = vtx2xy.shape().dims2()?;
        assert_eq!(two, 2);
        //
        let mut dw_vtx2xy = vec![0f32; num_vtx * 2];
        for i_elem in 0..self.elem2idx.len() - 1 {
            let num_vtx_in_elem = self.elem2idx[i_elem + 1] - self.elem2idx[i_elem];
            let ratio = if num_vtx_in_elem == 0 {
                0.0
            } else {
                1.0 / num_vtx_in_elem as f32
            };
            for i_edge in 0..num_vtx_in_elem {
                let i0_vtx = self.idx2vtx[self.elem2idx[i_elem] + i_edge];
                dw_vtx2xy[i0_vtx * 2] += ratio * dw_elem2cog[i_elem * 2];
                dw_vtx2xy[i0_vtx * 2 + 1] += ratio * dw_elem2cog[i_elem * 2 + 1];
            }
        }
        let dw_vtx2xy = candle_core::Tensor::from_vec(
            dw_vtx2xy,
            candle_core::Shape::from((num_vtx, 2)),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_vtx2xy))
    }
}
