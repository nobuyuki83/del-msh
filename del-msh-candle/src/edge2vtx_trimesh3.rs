#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
use candle_core::{CpuStorage, CustomOp1, Layout, Shape, Tensor};

pub struct Layer {
    tri2vtx: Tensor,
    edge2vtx: Tensor,
    edge2tri: Tensor,
    transform_world2ndc: Tensor,
}

impl CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        todo!()
    }

    fn cpu_fwd(
        &self,
        vtx2xyz: &CpuStorage,
        _l_vtx2xyz: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let vtx2xyz = vtx2xyz.as_slice::<f32>()?;
        use std::ops::Deref;
        get_cpu_slice_and_storage_from_tensor!(tri2vtx, s_tri2vtx, self.tri2vtx, u32);
        get_cpu_slice_and_storage_from_tensor!(edge2vtx, s_edge2vtx, self.edge2vtx, u32);
        get_cpu_slice_and_storage_from_tensor!(edge2tri, s_edge2tri, self.edge2tri, u32);
        get_cpu_slice_and_storage_from_tensor!(
            transform_world2ndc,
            s_transform_world2ndc,
            self.transform_world2ndc,
            f32
        );
        let transform_world2ndc = arrayref::array_ref![transform_world2ndc, 0, 16];
        let edge2vtx_contour = del_msh_core::edge2vtx::contour_for_triangle_mesh::<u32>(
            tri2vtx,
            vtx2xyz,
            transform_world2ndc,
            edge2vtx,
            edge2tri,
        );
        let num_edge_contour = edge2vtx_contour.len() / 2;
        Ok((
            CpuStorage::U32(edge2vtx_contour),
            (num_edge_contour, 2).into(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        vtx2xyz: &CudaStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DeviceSlice; // for CudaSlice::len
        use candle_core::cuda_backend::WrapErr;
        let device = &vtx2xyz.device;
        let num_edge = self.edge2vtx.dim(0)?;
        assert_eq!(self.edge2vtx.dim(1)?, 2);
        assert_eq!(l_vtx2xyz.dim(1)?, 3);
        use std::ops::Deref;
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            tri2vtx,
            s_tri2vtx,
            l_tri2vtx,
            self.tri2vtx,
            u32
        );
        assert_eq!(l_tri2vtx.dim(1)?, 3);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            edge2vtx,
            s_edge2vtx,
            l_edge2vtx,
            self.edge2vtx,
            u32
        );
        assert_eq!(l_edge2vtx.dim(1)?, 2);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            edge2tri,
            s_edge2tri,
            l_edge2tri,
            self.edge2tri,
            u32
        );
        assert_eq!(l_edge2tri.dims(), &[num_edge, 2]);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            transform_world2ndc,
            s_transform_world2ndc,
            l_transform_world2ndc,
            self.transform_world2ndc,
            f32
        );
        assert_eq!(l_transform_world2ndc.dims(), &[16]);
        let vtx2xyz = vtx2xyz.as_cuda_slice::<f32>()?;
        let edge2vtx_contour = del_msh_cudarc::edge2vtx_contour::fwd(
            device,
            tri2vtx,
            vtx2xyz,
            edge2vtx,
            edge2tri,
            transform_world2ndc,
        )
        .w()?;
        let num_edge_contour = edge2vtx_contour.len() / 2;
        let cuda_storage =
            candle_core::CudaStorage::wrap_cuda_slice(edge2vtx_contour, device.clone());
        Ok((cuda_storage, (num_edge_contour, 2).into()))
    }
}

#[test]
fn test_contour() -> candle_core::Result<()> {
    use candle_core::Device::Cpu;
    let (tri2vtx, vtx2xyz) =
        del_msh_core::trimesh3_primitive::torus_zup::<u32, f32>(1.0, 0.3, 32, 32);
    let num_tri = tri2vtx.len() / 3;
    let num_vtx = vtx2xyz.len() / 3;
    let edge2vtx = del_msh_core::edge2vtx::from_triangle_mesh(&tri2vtx, num_vtx);
    let num_edge = edge2vtx.len() / 2;
    let edge2tri = del_msh_core::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, num_vtx);
    //
    let img_asp = 1.5;
    let _img_shape = (((16 * 6) as f32 * img_asp) as usize, 16 * 6);
    let cam_projection =
        del_geo_core::mat4_col_major::camera_perspective_blender(img_asp, 24f32, 0.3, 10.0, true);
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.0], 0., 0., 0.);
    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat(&cam_projection, &cam_modelview);
    //
    let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &Cpu)?;
    let vtx2xyz = Tensor::from_vec(vtx2xyz, (num_vtx, 3), &Cpu)?;
    let edge2vtx = Tensor::from_vec(edge2vtx, (num_edge, 2), &Cpu)?;
    let edge2tri = Tensor::from_vec(edge2tri, (num_edge, 2), &Cpu)?;
    let transform_world2ndc = Tensor::from_slice(&transform_world2ndc, (16,), &Cpu)?;
    let layer = Layer {
        tri2vtx: tri2vtx.clone(),
        transform_world2ndc: transform_world2ndc.clone(),
        edge2vtx: edge2vtx.clone(),
        edge2tri: edge2tri.clone(),
    };
    let edge2vtx_contour = vtx2xyz.apply_op1_no_bwd(&layer)?;
    let edge2vtx_contour_cpu = edge2vtx_contour.flatten_all()?.to_vec1::<u32>()?;

    #[cfg(feature = "cuda")]
    {
        let device = candle_core::Device::cuda_if_available(0)?;
        let tri2vtx = tri2vtx.to_device(&device)?;
        let vtx2xyz = vtx2xyz.to_device(&device)?;
        let edge2vtx = edge2vtx.to_device(&device)?;
        let edge2tri = edge2tri.to_device(&device)?;
        let transform_world2ndc = transform_world2ndc.to_device(&device)?;
        let layer = Layer {
            tri2vtx: tri2vtx.clone(),
            transform_world2ndc: transform_world2ndc.clone(),
            edge2vtx: edge2vtx.clone(),
            edge2tri: edge2tri.clone(),
        };
        let edge2vtx_contour = vtx2xyz.apply_op1_no_bwd(&layer)?;
        let edge2vtx_contour_cuda = edge2vtx_contour.flatten_all()?.to_vec1::<u32>()?;
        assert_eq!(edge2vtx_contour_cuda.len(), edge2vtx_contour_cpu.len());
        edge2vtx_contour_cpu
            .iter()
            .zip(edge2vtx_contour_cuda.iter())
            .for_each(|(&a, &b)| assert_eq!(a, b));
    }

    Ok(())
}
