use candle_core::{CpuStorage, Device, Layout, Tensor};

struct Layer {
    pub tri2vtx: Tensor,
    pub vtx2xyz: Tensor,
    pub bvhnodes: Tensor,
}

impl candle_core::InplaceOp1 for Layer {
    fn name(&self) -> &'static str {
        "bvhnode2aabb"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        assert_eq!(layout.dims().len(), 2);
        let num_bvhnode = layout.dims()[0] as usize;
        assert_eq!(layout.dims()[1] as usize, 6);
        assert_eq!(self.bvhnodes.dims2()?.0, num_bvhnode);
        assert_eq!(self.bvhnodes.dims2()?.1, 3);
        use std::ops::Deref;
        let num_dim = self.vtx2xyz.dims2()?.1;
        let bvhnodes = self.bvhnodes.storage_and_layout().0;
        let bvhnodes = match bvhnodes.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let vtx2xyz = self.vtx2xyz.storage_and_layout().0;
        let vtx2xyz = match vtx2xyz.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let bvhnode2aabb = match storage {
            CpuStorage::F32(a) => a,
            _ => panic!(),
        };
        match num_dim {
            3 => del_msh_core::bvhnode2aabb3::update_for_uniform_mesh_with_bvh(
                bvhnode2aabb,
                0,
                bvhnodes,
                Some((tri2vtx, 3)),
                vtx2xyz,
                None,
            ),
            2 => del_msh_core::bvhnode2aabb2::update_for_uniform_mesh_with_bvh(
                bvhnode2aabb,
                0,
                bvhnodes,
                Some((tri2vtx, 3)),
                vtx2xyz,
                None,
            ),
            _ => panic!(),
        }
        Ok(())
    }
}

#[test]
fn test() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) =
        del_msh_core::trimesh3_primitive::torus_zup::<u32, f32>(1.0, 0.3, 32, 32);
    let (tri2vtx, vtx2xyz) = {
        let num_tri = tri2vtx.len() / 3;
        let tri2vtx =
            candle_core::Tensor::from_vec(tri2vtx, (num_tri, 3), &candle_core::Device::Cpu)?;
        let num_vtx = vtx2xyz.len() / 3;
        let vtx2xyz =
            candle_core::Tensor::from_vec(vtx2xyz, (num_vtx, 3), &candle_core::Device::Cpu)?;
        (tri2vtx, vtx2xyz)
    };
    let bvhnodes = crate::bvhnodes::from_trimesh(tri2vtx.clone(), vtx2xyz.clone())?;
    let layer = Layer {
        tri2vtx: tri2vtx.clone(),
        vtx2xyz: vtx2xyz.clone(),
        bvhnodes: bvhnodes.clone(),
    };
    let num_bvhnode = bvhnodes.dims2()?.0;
    let bvhnode2aabbiii = Tensor::zeros((num_bvhnode, 6), candle_core::DType::F32, &Device::Cpu)?;
    bvhnode2aabbiii.inplace_op1(&layer)?;
    Ok(())
}
