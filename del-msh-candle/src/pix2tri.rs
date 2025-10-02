use candle_core::{CpuStorage, Layout, Tensor};
use std::ops::Deref;

#[cfg(feature = "cuda")]
use candle_core::{backend::BackendDevice, CudaStorage};

pub struct Pix2Tri {
    pub bvhnodes: Tensor,
    pub bvhnode2aabb: Tensor,
    pub transform_ndc2world: Tensor,
}

impl candle_core::InplaceOp3 for Pix2Tri {
    fn name(&self) -> &'static str {
        "pix2tri"
    }
    fn cpu_fwd(
        &self,
        pix2tri: &mut CpuStorage,
        l_pix2tri: &Layout,
        tri2vtx: &CpuStorage,
        l_tri2vtx: &Layout,
        vtx2xyz: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        assert_eq!(l_tri2vtx.dim(1)?, 3);
        let num_tri = l_tri2vtx.dim(0)?;
        let num_dim = l_vtx2xyz.dim(1)?;
        assert_eq!(num_dim, 3); // todo: implement num_dim == 2
        assert_eq!(self.bvhnodes.dims2()?, (num_tri * 2 - 1, 3));
        assert_eq!(self.bvhnode2aabb.dims2()?, (num_tri * 2 - 1, 6));
        let img_shape = (l_pix2tri.dim(1)?, l_pix2tri.dim(0)?);
        let pix2tri = match pix2tri {
            CpuStorage::U32(v) => v,
            _ => panic!(),
        };
        let tri2vtx = tri2vtx.as_slice()?;
        let vtx2xyz = vtx2xyz.as_slice()?;
        get_cpu_slice_and_storage_from_tensor!(bvhnodes, l_bvhnodes, self.bvhnodes, u32);
        get_cpu_slice_and_storage_from_tensor!(
            bvhnode2aabb,
            l_bvhnode2aabb,
            self.bvhnode2aabb,
            f32
        );
        get_cpu_slice_and_storage_from_tensor!(
            transform_ndc2world,
            l_transform_ndc2world,
            self.transform_ndc2world,
            f32
        );
        let transform_ndc2world = arrayref::array_ref!(transform_ndc2world, 0, 16);
        del_msh_cpu::trimesh3_raycast::update_pix2tri(
            pix2tri,
            tri2vtx,
            vtx2xyz,
            bvhnodes,
            bvhnode2aabb,
            img_shape,
            transform_ndc2world,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        pix2tri: &mut CudaStorage,
        l_pix2tri: &Layout,
        tri2vtx: &CudaStorage,
        l_tri2vtx: &Layout,
        vtx2xyz: &CudaStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::cuda::CudaStorageSlice;
        assert_eq!(l_tri2vtx.dim(1)?, 3);
        assert_eq!(l_vtx2xyz.dim(1)?, 3); // todo: implement 2D
        use candle_core::cuda_backend::WrapErr;
        let img_shape = (l_pix2tri.dim(1)?, l_pix2tri.dim(0)?);
        get_cuda_slice_device_from_storage_u32!(pix2tri, device_pix2tri, pix2tri);
        get_cuda_slice_device_from_storage_u32!(tri2vtx, device_tri2vtx, tri2vtx);
        get_cuda_slice_device_from_storage_f32!(vtx2xyz, device_vtx2xyz, vtx2xyz);
        assert!(device_pix2tri.same_device(device_tri2vtx));
        assert!(device_pix2tri.same_device(device_vtx2xyz));
        //
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            bvhnodes,
            _storage,
            _layout,
            self.bvhnodes,
            u32
        );
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            bvhnode2aabb,
            _storage,
            _layout,
            self.bvhnode2aabb,
            f32
        );
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            transform_ndc2world,
            _storage,
            _layout,
            self.transform_ndc2world,
            f32
        );
        del_msh_cudarc_safe::pix2tri::fwd(
            &device_pix2tri.cuda_stream(),
            img_shape,
            pix2tri,
            tri2vtx,
            vtx2xyz,
            bvhnodes,
            bvhnode2aabb,
            transform_ndc2world,
        )
        .w()?;
        Ok(())
    }
}

pub fn from_trimesh3(
    tri2vtx: &Tensor,
    vtx2xyz: &Tensor,
    bvhnodes: &Tensor,
    bvhnode2aabb: &Tensor,
    img_shape: (usize, usize),    // (width, height)
    transform_ndc2world: &Tensor, // transform column major
) -> candle_core::Result<Tensor> {
    let device = tri2vtx.device();
    let pix2tri = Tensor::zeros((img_shape.1, img_shape.0), candle_core::DType::U32, device)?;
    let layer = Pix2Tri {
        bvhnodes: bvhnodes.clone(),
        bvhnode2aabb: bvhnode2aabb.clone(),
        transform_ndc2world: transform_ndc2world.clone(),
    };
    pix2tri.inplace_op3(tri2vtx, vtx2xyz, &layer)?;
    Ok(pix2tri)
}
