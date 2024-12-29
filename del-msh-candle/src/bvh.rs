use std::ops::Deref;

pub fn from_trimesh2(
    tri2vtx: &candle_core::Tensor,
    vtx2xy: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>()?,
        _ => panic!(),
    };
    let vtx2xy = vtx2xy.storage_and_layout().0;
    let vtx2xy = match vtx2xy.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh::<u32>(tri2vtx, vtx2xy, 2);
    let aabbs = del_msh_core::bvhnode2aabb2::from_uniform_mesh_with_bvh::<u32, f32>(
        0,
        &bvhnodes,
        Some((tri2vtx, 3)),
        vtx2xy,
        None,
    );
    let num_bvhnode = bvhnodes.len() / 3;
    let bvhnodes = candle_core::Tensor::from_vec(
        bvhnodes,
        candle_core::Shape::from((num_bvhnode, 3)),
        &candle_core::Device::Cpu,
    )?;
    let num_aabb = aabbs.len() / 4;
    let aabbs = candle_core::Tensor::from_vec(
        aabbs,
        candle_core::Shape::from((num_aabb, 4)),
        &candle_core::Device::Cpu,
    )?;
    Ok((bvhnodes, aabbs))
}

fn from_trimesh3_cpu(
    tri2vtx: &candle_core::Tensor,
    vtx2xyz: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>()?,
        _ => panic!(),
    };
    let vtx2xyz = vtx2xyz.storage_and_layout().0;
    let vtx2xyz = match vtx2xyz.deref() {
        candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
        _ => panic!(),
    };
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh::<u32>(tri2vtx, vtx2xyz, 3);
    let aabbs = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh::<u32, f32>(
        0,
        &bvhnodes,
        Some((tri2vtx, 3)),
        vtx2xyz,
        None,
    );
    let num_bvhnode = bvhnodes.len() / 3;
    let bvhnodes = candle_core::Tensor::from_vec(
        bvhnodes,
        candle_core::Shape::from((num_bvhnode, 3)),
        &candle_core::Device::Cpu,
    )?;
    let num_aabb = aabbs.len() / 6;
    let aabbs = candle_core::Tensor::from_vec(
        aabbs,
        candle_core::Shape::from((num_aabb, 6)),
        &candle_core::Device::Cpu,
    )?;
    Ok((bvhnodes, aabbs))
}

#[cfg(feature = "cuda")]
fn from_trimesh3_cuda(
    tri2vtx: &candle_core::Tensor,
    vtx2xyz: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    let dev = tri2vtx.device().as_cuda_device()?;
    use candle_core::backend::BackendDevice;
    assert!(dev.same_device(vtx2xyz.device().as_cuda_device()?));
    let num_tri = tri2vtx.shape().dims2()?.0;
    use std::ops::Deref;
    let tri2vtx = tri2vtx.storage_and_layout().0;
    let tri2vtx = match tri2vtx.deref() {
        candle_core::Storage::Cuda(cpu_tri2vtx) => cpu_tri2vtx.as_cuda_slice::<u32>(),
        _ => panic!(),
    }?;
    let vtx2xyz = vtx2xyz.storage_and_layout().0;
    let vtx2xyz = match vtx2xyz.deref() {
        candle_core::Storage::Cuda(cpu_vtx2xyz) => cpu_vtx2xyz.as_cuda_slice::<f32>(),
        _ => panic!(),
    }?;
    // let mut tri2cntr = dev.alloc_zeros::<f32>(num_tri*3)?;
    // del_cudarc_bvh::bvh::tri2cntr_from_trimesh3(dev, tri2vtx, vtx2xyz, &mut tri2cntr)?;
    let num_bvh = num_tri * 2 - 1;
    let bvhnodes = candle_core::Tensor::zeros(
        (num_bvh, 3),
        candle_core::DType::U32,
        &candle_core::Device::Cuda(dev.clone()),
    )?;
    let bvhnode2aabb = candle_core::Tensor::zeros(
        (num_bvh, 6),
        candle_core::DType::F32,
        &candle_core::Device::Cuda(dev.clone()),
    )?;
    {
        let bvhnodes = bvhnodes.storage_and_layout().0;
        let bvhnodes = match bvhnodes.deref() {
            candle_core::Storage::Cuda(bvhnodes) => bvhnodes.as_cuda_slice::<u32>(),
            _ => panic!(),
        }?;
        let bvhnode2aabb = bvhnode2aabb.storage_and_layout().0;
        let bvhnode2aabb = match bvhnode2aabb.deref() {
            candle_core::Storage::Cuda(bvhnode2aabb) => bvhnode2aabb.as_cuda_slice::<f32>(),
            _ => panic!(),
        }?;
        let now = std::time::Instant::now();
        match del_msh_cudarc::make_bvh_from_trimesh3(
            dev,
            &bvhnodes,
            &bvhnode2aabb,
            tri2vtx,
            vtx2xyz,
        ) {
            Err(e) => {
                return Err(candle_core::Error::Msg(e.to_string()));
            }
            Ok(()) => {}
        }
        println!("{:?}", now.elapsed());
    }
    // use candle_core::cuda_backend::cudarc::driver::DeviceSlice;
    Ok((bvhnodes, bvhnode2aabb))
}

pub fn from_trimesh3(
    tri2vtx: &candle_core::Tensor,
    vtx2xyz: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    use candle_core::Device::{Cpu, Cuda};
    match (tri2vtx.device(), vtx2xyz.device()) {
        (&Cpu, &Cpu) => from_trimesh3_cpu(tri2vtx, vtx2xyz),
        (Cuda(c), Cuda(b)) => {
            use candle_core::backend::BackendDevice;
            assert!(c.same_device(b));
            #[cfg(feature = "cuda")]
            return from_trimesh3_cuda(tri2vtx, vtx2xyz);
            #[cfg(not(feature = "cuda"))]
            return Err(candle_core::Error::NotCompiledWithCudaSupport);
        }
        _ => Err(candle_core::Error::Msg("hogehoge".to_string())),
    }
}

#[test]
fn test_from_trimesh() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz) =
        del_msh_core::trimesh3_primitive::torus_zup::<u32, f32>(1.0, 0.3, 32, 32);
    let num_tri = tri2vtx.len() / 3;
    let tri2vtx = candle_core::Tensor::from_vec(tri2vtx, (num_tri, 3), &candle_core::Device::Cpu)?;
    let num_vtx = vtx2xyz.len() / 3;
    let vtx2xyz = candle_core::Tensor::from_vec(vtx2xyz, (num_vtx, 3), &candle_core::Device::Cpu)?;
    let (bvhnodes_cpu, bvhnode2aabb_cpu) = {
        let bvh = from_trimesh3(&tri2vtx, &vtx2xyz)?;
        (
            bvh.0.flatten_all()?.to_vec1::<u32>()?,
            bvh.1.flatten_all()?.to_vec1::<f32>()?,
        )
    };
    #[cfg(feature = "cuda")]
    {
        let (bvhnodes_gpu, bvhnode2aabb_gpu) = {
            let device = candle_core::Device::new_cuda(0)?;
            let tri2vtx = tri2vtx.to_device(&device)?;
            let vtx2xyz = vtx2xyz.to_device(&device)?;
            let bvh = from_trimesh3(&tri2vtx, &vtx2xyz)?;
            (
                bvh.0.flatten_all()?.to_vec1::<u32>()?,
                bvh.1.flatten_all()?.to_vec1::<f32>()?,
            )
        };
        assert_eq!(bvhnodes_cpu.len(), bvhnodes_gpu.len());
        bvhnodes_cpu
            .iter()
            .zip(bvhnodes_gpu.iter())
            .for_each(|(&a, &b)| assert_eq!(a, b));
        assert_eq!(bvhnode2aabb_cpu.len(), bvhnode2aabb_gpu.len());
        bvhnode2aabb_cpu
            .iter()
            .zip(bvhnode2aabb_gpu.iter())
            .for_each(|(&a, &b)| assert_eq!(a, b));
    }
    Ok(())
}
