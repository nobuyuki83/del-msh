use candle_core::{CpuStorage, CudaStorage, Device::Cpu, Layout, Result, Tensor};

struct FromTrimesh_UsingMorton {
    pub tri2vtx: Tensor,
    pub vtx2xyz: Tensor,
}

impl candle_core::InplaceOp1 for FromTrimesh_UsingMorton {
    fn name(&self) -> &'static str {
        "bvhnodes"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> Result<()> {
        use std::ops::Deref;
        assert_eq!(layout.dims().len(), 2);
        assert_eq!(layout.dims()[1], 3);
        assert_eq!(self.tri2vtx.dims2()?.1, 3);
        let num_tri = self.tri2vtx.dims2()?.0;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>()?,
            _ => panic!(),
        };
        let num_dim = self.vtx2xyz.dims2()?.1;
        assert!(num_dim == 2 || num_dim == 3);
        let vtx2xyz = self.vtx2xyz.storage_and_layout().0;
        let vtx2xyz = match vtx2xyz.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let bvhnodes = match storage {
            CpuStorage::U32(s) => s,
            _ => panic!(),
        };
        del_msh_core::bvhnodes_morton::from_triangle_mesh_inplace(
            bvhnodes, tri2vtx, vtx2xyz, num_dim,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, sto: &mut CudaStorage, layout: &Layout) -> Result<()> {
        Ok(())
    }
}

pub fn from_trimesh(tri2vtx: Tensor, vtx2xyz: Tensor) -> Result<Tensor> {
    let num_tri = tri2vtx.dims2()?.0;
    let layer = FromTrimesh_UsingMorton { tri2vtx, vtx2xyz };
    let bvhnodes = Tensor::zeros((num_tri * 2 - 1, 3), candle_core::DType::U32, &Cpu)?;
    bvhnodes.inplace_op1(&layer)?;
    Ok(bvhnodes)
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
    let num_tri = tri2vtx.dims2()?.0;
    let bvhnodes = from_trimesh(tri2vtx, vtx2xyz)?;
    {
        let bvhnodes = bvhnodes.flatten_all()?.to_vec1::<u32>()?;
        del_msh_core::bvhnodes::check_bvh_topology(&bvhnodes, num_tri);
    }
    Ok(())
}
