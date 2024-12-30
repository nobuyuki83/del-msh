#[allow(unused_imports)]
use candle_core::{CpuStorage, CudaDevice, CudaStorage, DType, Device, Layout, Tensor};

pub struct Layer {
    pub tri2vtx: Tensor,
}

impl candle_core::InplaceOp3 for Layer {
    fn name(&self) -> &'static str {
        "bvhnode2aabb"
    }

    fn cpu_fwd(
        &self,
        bvhnode2aabb: &mut CpuStorage,
        l_bvhnode2aabb: &Layout,
        bvhnodes: &CpuStorage,
        l_bvhnodes: &Layout,
        vtx2xyz: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        assert_eq!(l_bvhnode2aabb.dims().len(), 2);
        let num_bvhnode = l_bvhnode2aabb.dims()[0];
        let num_dim = l_vtx2xyz.dim(1)?;
        assert_eq!(l_bvhnode2aabb.dim(1)?, num_dim * 2);
        assert_eq!(l_bvhnodes.dim(0)?, num_bvhnode);
        assert_eq!(l_bvhnodes.dim(1)?, 3);
        use std::ops::Deref;
        let bvhnodes = match bvhnodes {
            CpuStorage::U32(a) => a,
            _ => panic!(),
        };
        let vtx2xyz = match vtx2xyz {
            CpuStorage::F32(a) => a,
            _ => panic!(),
        };
        let bvhnode2aabb = match bvhnode2aabb {
            CpuStorage::F32(a) => a,
            _ => panic!(),
        };
        let num_node = self.tri2vtx.dims2()?.1;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        match num_dim {
            3 => del_msh_core::bvhnode2aabb3::update_for_uniform_mesh_with_bvh(
                bvhnode2aabb,
                0,
                bvhnodes,
                Some((tri2vtx, num_node)),
                vtx2xyz,
                None,
            ),
            2 => del_msh_core::bvhnode2aabb2::update_for_uniform_mesh_with_bvh(
                bvhnode2aabb,
                0,
                bvhnodes,
                Some((tri2vtx, num_node)),
                vtx2xyz,
                None,
            ),
            _ => panic!(),
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        bvhnode2aabb: &mut CudaStorage,
        l_bvhnode2aabb: &Layout,
        bvhnodes: &CudaStorage,
        l_bvhnodes: &Layout,
        vtx2xyz: &CudaStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        let num_tri = self.tri2vtx.storage_and_layout().1.dim(0)?;
        let num_bvhnode = num_tri * 2 - 1;
        assert_eq!(l_bvhnode2aabb.dims(), &[num_bvhnode, 6]);
        assert_eq!(l_bvhnodes.dims(), &[num_bvhnode, 3]);
        assert_eq!(l_vtx2xyz.dim(1)?, 3);
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        use std::ops::Deref;
        let CudaStorage { slice, device } = bvhnode2aabb;
        let (bvhnode2aabb, device_bvhnodes2aabb) = match slice {
            CudaStorageSlice::F32(slice) => (slice, device),
            _ => panic!(),
        };
        //
        let CudaStorage { slice, device } = vtx2xyz;
        let (vtx2xyz, device_vtx2xyz) = match slice {
            CudaStorageSlice::F32(slice) => (slice, device),
            _ => panic!(),
        };
        //
        let CudaStorage { slice, device } = bvhnodes;
        let (bvhnodes, device_bvhnodes) = match slice {
            CudaStorageSlice::U32(slice) => (slice, device),
            _ => panic!(),
        };
        assert!(device_bvhnodes2aabb.same_device(device_vtx2xyz));
        assert!(device_bvhnodes2aabb.same_device(device_bvhnodes));
        //
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice()?,
            _ => panic!(),
        };
        del_msh_cudarc::bvhnode2aabb::from_trimesh3_with_bvhnodes(
            device,
            tri2vtx,
            vtx2xyz,
            &bvhnodes.slice(..),
            &mut bvhnode2aabb.slice_mut(..),
        )
        .w()?;
        Ok(())
    }
}

pub struct BvhForTriMesh {
    pub tri2center: Tensor,
    pub sorted_morton_code: Tensor,
    pub bvhnodes: Tensor,
    pub bvhnode2aabb: Tensor,
}

impl BvhForTriMesh {
    pub fn new(
        num_tri: usize,
        num_dim: usize,
        device: &Device,
    ) -> std::result::Result<Self, candle_core::Error> {
        let tri2center = candle_core::Tensor::zeros((num_tri, num_dim), DType::F32, device)?;
        let sorted_morton_code = candle_core::Tensor::zeros(num_tri * 3, DType::U32, device)?;
        let num_bvhnodes = num_tri * 2 - 1;
        let bvhnodes = candle_core::Tensor::zeros((num_bvhnodes, 3), DType::U32, device)?;
        let bvhnode2aabb =
            candle_core::Tensor::zeros((num_bvhnodes, num_dim * 2), DType::F32, device)?;
        let res = BvhForTriMesh {
            tri2center,
            sorted_morton_code,
            bvhnodes,
            bvhnode2aabb,
        };
        Ok(res)
    }

    pub fn compute(&self, tri2vtx: &Tensor, vtx2xyz: &Tensor) -> candle_core::Result<()> {
        self.tri2center
            .inplace_op3(tri2vtx, vtx2xyz, &crate::elem2center::Layer {})?;
        self.sorted_morton_code.inplace_op2(
            &self.tri2center,
            &crate::bvhnodes_morton::SortedMortonCode {},
        )?;
        self.bvhnodes.inplace_op2(
            &self.sorted_morton_code,
            &crate::bvhnodes_morton::BvhNodesFromSortedMortonCode {},
        )?;
        let layer = Layer {
            tri2vtx: tri2vtx.clone(),
        };
        self.bvhnode2aabb
            .inplace_op3(&self.bvhnodes, vtx2xyz, &layer)?;
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
    let num_tri = tri2vtx.dims2()?.0;
    let bvhdata = BvhForTriMesh::new(num_tri, 3, &Device::Cpu)?;
    bvhdata.compute(&tri2vtx, &vtx2xyz)?;
    let bvhnode2aabb_cpu = bvhdata.bvhnode2aabb.flatten_all()?.to_vec1::<f32>()?;
    #[cfg(feature = "cuda")]
    {
        let device = Device::new_cuda(0)?;
        let bvhdata = BvhForTriMesh::new(num_tri, 3, &device)?;
        let tri2vtx = tri2vtx.to_device(&device)?;
        let vtx2xyz = vtx2xyz.to_device(&device)?;
        bvhdata.compute(&tri2vtx, &vtx2xyz)?;
        let bvhnode2aabb_gpu = bvhdata.bvhnode2aabb.flatten_all()?.to_vec1::<f32>()?;
        bvhnode2aabb_cpu
            .iter()
            .zip(bvhnode2aabb_gpu.iter())
            .for_each(|(a, b)| {
                assert_eq!(a, b);
            })
    }
    Ok(())
}
