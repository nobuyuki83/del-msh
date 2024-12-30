#[allow(unused_imports)]
use candle_core::{CpuStorage, CudaStorage, Layout};

pub struct Layer {}

impl candle_core::InplaceOp3 for Layer {
    fn name(&self) -> &'static str {
        "elem2center"
    }

    fn cpu_fwd(
        &self,
        elem2center: &mut CpuStorage,
        l_elem2center: &Layout,
        elem2vtx: &CpuStorage,
        l_elem2vtx: &Layout,
        vtx2pos: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        let num_elem = l_elem2vtx.dim(0)?;
        let num_node = l_elem2vtx.dim(1)?;
        let num_dim = l_vtx2xyz.dim(1)?;
        assert_eq!(l_elem2center.dims(), &[num_elem, num_dim]);
        let elem2vtx = match elem2vtx {
            CpuStorage::U32(vec) => vec,
            _ => panic!(),
        };
        let vtx2xyz = match vtx2pos {
            CpuStorage::F32(vec) => vec,
            _ => panic!(),
        };
        let elem2center = match elem2center {
            CpuStorage::F32(vec) => vec,
            _ => panic!(),
        };
        del_msh_core::elem2center::update_from_uniform_mesh_as_points(
            elem2center,
            elem2vtx,
            num_node,
            vtx2xyz,
            num_dim,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        elem2center: &mut CudaStorage,
        l_elem2center: &Layout,
        elem2vtx: &CudaStorage,
        l_elem2vtx: &Layout,
        vtx2pos: &CudaStorage,
        l_vtx2pos: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        let num_elem = l_elem2vtx.dim(0)?;
        let _num_node = l_elem2vtx.dim(1)?;
        let num_dim = l_vtx2pos.dim(1)?;
        assert_eq!(l_elem2center.dims(), &[num_elem, num_dim]);
        let CudaStorage { slice, device } = elem2vtx;
        let (elem2vtx, device_elem2vtx) = match slice {
            CudaStorageSlice::U32(slice) => (slice, device),
            _ => todo!(),
        };
        let CudaStorage { slice, device } = vtx2pos;
        let (vtx2pos, dev_vtx2pos) = match slice {
            CudaStorageSlice::F32(slice) => (slice, device),
            _ => panic!(),
        };
        let CudaStorage { slice, device } = elem2center;
        let (elem2center, dev_elem2center) = match slice {
            CudaStorageSlice::F32(slice) => (slice, device),
            _ => panic!(),
        };
        assert!(device_elem2vtx.same_device(dev_vtx2pos));
        assert!(device_elem2vtx.same_device(dev_elem2center));
        del_msh_cudarc::elem2center::tri2cntr_from_trimesh3(
            device_elem2vtx,
            elem2vtx,
            vtx2pos,
            elem2center,
        )
        .w()?;
        Ok(())
    }
}

#[test]
fn test() -> anyhow::Result<()> {
    use candle_core::{Device, Tensor};
    let (tri2vtx, vtx2xyz) =
        del_msh_core::trimesh3_primitive::torus_zup::<u32, f32>(1.0, 0.3, 32, 32);
    let (tri2vtx, vtx2xyz) = {
        let num_tri = tri2vtx.len() / 3;
        let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &Device::Cpu)?;
        let num_vtx = vtx2xyz.len() / 3;
        let vtx2xyz = Tensor::from_vec(vtx2xyz, (num_vtx, 3), &Device::Cpu)?;
        (tri2vtx, vtx2xyz)
    };
    let num_tri = tri2vtx.dims2()?.0;
    let elem2center_cpu = {
        let elem2center = Tensor::zeros(
            (num_tri, 3),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )?;
        elem2center.inplace_op3(&tri2vtx, &vtx2xyz, &Layer {})?;
        elem2center.flatten_all()?.to_vec1::<f32>()?
    };
    #[cfg(feature = "cuda")]
    {
        let device = Device::new_cuda(0)?;
        let tri2vtx = tri2vtx.to_device(&device)?;
        let vtx2xyz = vtx2xyz.to_device(&device)?;
        let elem2center = Tensor::zeros((num_tri, 3), candle_core::DType::F32, &device)?;
        elem2center.inplace_op3(&tri2vtx, &vtx2xyz, &Layer {})?;
        let elem2center_gpu = elem2center.flatten_all()?.to_vec1::<f32>()?;
        elem2center_cpu
            .iter()
            .zip(elem2center_gpu.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            });
    };
    Ok(())
}
