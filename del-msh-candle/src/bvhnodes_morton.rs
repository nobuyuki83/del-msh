#[allow(unused_imports)]
use candle_core::{CpuStorage, CudaStorage, Device, Layout, Tensor};

pub struct SortedMortonCode {}

impl candle_core::InplaceOp2 for SortedMortonCode {
    fn name(&self) -> &'static str {
        "bvhnodes_morton"
    }
    fn cpu_fwd(
        &self,
        morton_data: &mut CpuStorage,
        _l1: &Layout,
        vtx2pos: &CpuStorage,
        l_vtx2pos: &Layout,
    ) -> candle_core::Result<()> {
        let num_vtx = l_vtx2pos.dims()[0];
        let num_dim = l_vtx2pos.dims()[1];
        let vtx2pos = vtx2pos.as_slice::<f32>()?;
        let morton_data = match morton_data {
            CpuStorage::U32(morton_data) => morton_data,
            _ => panic!(),
        };
        assert_eq!(morton_data.len(), num_vtx * 3);
        let (idx2vtx, idx2morton) = morton_data.split_at_mut(num_vtx);
        let (idx2morton, vtx2morton) = idx2morton.split_at_mut(num_vtx);
        del_msh_core::bvhnodes_morton::update_sorted_morton_code(
            idx2vtx, idx2morton, vtx2morton, vtx2pos, num_dim,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        sorted_morton_code: &mut CudaStorage,
        _l_sorted_morton_code: &Layout,
        vtx2pos: &CudaStorage,
        l_vtx2pos: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        get_cuda_slice_from_storage_u32!(
            sorted_morton_code,
            device_sorted_morton_code,
            sorted_morton_code
        );
        get_cuda_slice_from_storage_f32!(vtx2pos, device_vtx2pos, vtx2pos);
        assert!(device_sorted_morton_code.same_device(device_vtx2pos));
        let aabb = del_msh_cudarc::vtx2xyz::to_aabb3(device_sorted_morton_code, vtx2pos).w()?;
        let transform_cntr2uni = {
            let aabb_cpu = device_sorted_morton_code.dtoh_sync_copy(&aabb).w()?;
            let aabb_cpu = arrayref::array_ref!(&aabb_cpu, 0, 6);
            let transform_cntr2uni_cpu =
                del_geo_core::mat4_col_major::from_aabb3_fit_into_unit_preserve_asp(aabb_cpu);
            device_sorted_morton_code
                .htod_copy(transform_cntr2uni_cpu.to_vec())
                .w()?
        };
        let num_vtx = l_vtx2pos.dim(0)?;
        let (mut idx2vtx, mut idx2morton) = sorted_morton_code.split_at_mut(num_vtx);
        let (mut idx2morton, mut vtx2morton) = idx2morton.split_at_mut(num_vtx);
        del_msh_cudarc::bvhnodes_morton::vtx2morton(
            device_sorted_morton_code,
            vtx2pos,
            &transform_cntr2uni,
            &mut vtx2morton,
        )
        .w()?;
        del_cudarc::util::set_consecutive_sequence(device_sorted_morton_code, &mut idx2vtx).w()?;
        del_cudarc::sort_by_key_u32::radix_sort_by_key_u32(
            device_sorted_morton_code,
            &mut vtx2morton,
            &mut idx2vtx,
        )
        .w()?;
        device_sorted_morton_code
            .dtod_copy(&vtx2morton, &mut idx2morton)
            .w()?;
        Ok(())
    }
}

pub struct BvhNodesFromSortedMortonCode {}

impl candle_core::InplaceOp2 for BvhNodesFromSortedMortonCode {
    fn name(&self) -> &'static str {
        "BvhNodesFromSortedMorton"
    }

    fn cpu_fwd(
        &self,
        bvhnodes: &mut CpuStorage,
        l_bvhnodes: &Layout,
        morton_data: &CpuStorage,
        l_morton_data: &Layout,
    ) -> candle_core::Result<()> {
        let num_vtx = l_morton_data.dims()[0] / 3;
        assert_eq!(l_bvhnodes.dims()[1], 3);
        assert_eq!(l_bvhnodes.dims()[0], num_vtx * 2 - 1);
        let bvhnodes = match bvhnodes {
            CpuStorage::U32(bvhnodes) => bvhnodes,
            _ => panic!(),
        };
        let morton_data = morton_data.as_slice::<u32>()?;
        let (idx2vtx, idx2morton) = morton_data.split_at(num_vtx);
        let (idx2morton, _vtx2morton) = idx2morton.split_at(num_vtx);
        del_msh_core::bvhnodes_morton::update_bvhnodes(bvhnodes, idx2vtx, idx2morton);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        bvhnodes: &mut CudaStorage,
        l_bvhnodes: &Layout,
        sorted_morton_code: &CudaStorage,
        l_morton_data: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        get_cuda_slice_from_storage_u32!(bvhnodes, device_bvhnodes, bvhnodes);
        let num_vtx = l_morton_data.dim(0)? / 3;
        assert_eq!(l_bvhnodes.dims(), &[num_vtx * 2 - 1, 3]);
        get_cuda_slice_from_storage_u32!(
            sorted_morton_code,
            device_sorted_morton_code,
            sorted_morton_code
        );
        assert!(device_bvhnodes.same_device(device_sorted_morton_code));
        let idx2vtx = sorted_morton_code.slice(0..num_vtx);
        let idx2morton = sorted_morton_code.slice(num_vtx..num_vtx * 2);
        del_msh_cudarc::bvhnodes_morton::from_sorted_morton_codes(
            device_bvhnodes,
            &mut bvhnodes.slice_mut(0..),
            &idx2morton,
            &idx2vtx,
        )
        .w()?;
        Ok(())
    }
}

// ---------------------------

pub fn from_trimesh(tri2vtx: &Tensor, vtx2xyz: &Tensor) -> anyhow::Result<Tensor> {
    let num_tri = tri2vtx.dims2()?.0;
    let tri2center = candle_core::Tensor::zeros(
        (num_tri, 3),
        candle_core::DType::F32,
        &candle_core::Device::Cpu,
    )?;
    let sorted_morton_code = candle_core::Tensor::zeros(
        num_tri * 3,
        candle_core::DType::U32,
        &candle_core::Device::Cpu,
    )?;
    let bvhnodes = candle_core::Tensor::zeros(
        (num_tri * 2 - 1, 3),
        candle_core::DType::U32,
        &candle_core::Device::Cpu,
    )?;
    // -----------------
    tri2center.inplace_op3(tri2vtx, vtx2xyz, &crate::elem2center::Layer {})?;
    sorted_morton_code.inplace_op2(&tri2center, &SortedMortonCode {})?;
    bvhnodes.inplace_op2(&sorted_morton_code, &BvhNodesFromSortedMortonCode {})?;
    Ok(bvhnodes)
}

#[test]
fn test_from_vtx2xyz() -> anyhow::Result<()> {
    let num_vtx = 1000;
    let vtx2xyz = {
        let vtx2xyz: Vec<f32> = {
            use rand::Rng;
            let mut rng = rand::rng();
            (0..num_vtx * 3).map(|_| rng.random::<f32>()).collect()
        };
        candle_core::Tensor::from_vec(vtx2xyz, (num_vtx, 3), &candle_core::Device::Cpu)?
    };
    let sorted_morton_code = candle_core::Tensor::zeros(
        num_vtx * 3,
        candle_core::DType::U32,
        &candle_core::Device::Cpu,
    )?;
    let bvhnodes = candle_core::Tensor::zeros(
        (num_vtx * 2 - 1, 3),
        candle_core::DType::U32,
        &candle_core::Device::Cpu,
    )?;
    sorted_morton_code.inplace_op2(&vtx2xyz, &SortedMortonCode {})?;
    bvhnodes.inplace_op2(&sorted_morton_code, &BvhNodesFromSortedMortonCode {})?;
    {
        let bvhnodes = bvhnodes.flatten_all()?.to_vec1::<u32>()?;
        del_msh_core::bvhnodes::check_bvh_topology(&bvhnodes, num_vtx);
    }
    Ok(())
}

#[test]
#[allow(unused_variables)]
fn test_from_trimesh3() -> anyhow::Result<()> {
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
    let tri2center = candle_core::Tensor::zeros(
        (num_tri, 3),
        candle_core::DType::F32,
        &candle_core::Device::Cpu,
    )?;
    let sorted_morton_code = candle_core::Tensor::zeros(
        num_tri * 3,
        candle_core::DType::U32,
        &candle_core::Device::Cpu,
    )?;
    let bvhnodes = candle_core::Tensor::zeros(
        (num_tri * 2 - 1, 3),
        candle_core::DType::U32,
        &candle_core::Device::Cpu,
    )?;
    // -----------------
    tri2center.inplace_op3(&tri2vtx, &vtx2xyz, &crate::elem2center::Layer {})?;
    let tri2center_cpu = tri2center.flatten_all()?.to_vec1::<f32>()?;
    sorted_morton_code.inplace_op2(&tri2center, &SortedMortonCode {})?;
    let sorted_morton_code_cpu = sorted_morton_code.flatten_all()?.to_vec1::<u32>()?;
    bvhnodes.inplace_op2(&sorted_morton_code, &BvhNodesFromSortedMortonCode {})?;
    let bvhnodes_cpu = bvhnodes.flatten_all()?.to_vec1::<u32>()?;
    #[cfg(feature = "cuda")]
    {
        let device = Device::new_cuda(0)?;
        let tri2vtx = tri2vtx.to_device(&device)?;
        let vtx2xyz = vtx2xyz.to_device(&device)?;
        let tri2center = tri2center.zeros_like()?.to_device(&device)?;
        tri2center.inplace_op3(&tri2vtx, &vtx2xyz, &crate::elem2center::Layer {})?;
        let tri2center_gpu = tri2center.flatten_all()?.to_vec1::<f32>()?;
        tri2center_cpu
            .iter()
            .zip(tri2center_gpu.iter())
            .for_each(|(&a, &b)| {
                assert_eq!(a, b);
            });
        let sorted_morton_code = sorted_morton_code.zeros_like()?.to_device(&device)?;
        sorted_morton_code.inplace_op2(&tri2center, &SortedMortonCode {})?;
        let sorted_morton_code_gpu = sorted_morton_code.flatten_all()?.to_vec1::<u32>()?;
        sorted_morton_code_cpu
            .iter()
            .zip(sorted_morton_code_gpu.iter())
            .take(num_tri * 2)
            .for_each(|(&a, &b)| {
                assert_eq!(a, b);
            });
        let bvhnodes = bvhnodes.zeros_like()?.to_device(&device)?;
        bvhnodes.inplace_op2(&sorted_morton_code, &BvhNodesFromSortedMortonCode {})?;
        let bvhnodes_gpu = bvhnodes.flatten_all()?.to_vec1::<u32>()?;
        bvhnodes_cpu
            .iter()
            .zip(bvhnodes_gpu.iter())
            .for_each(|(&a, &b)| assert_eq!(a, b));
    }
    Ok(())
}
