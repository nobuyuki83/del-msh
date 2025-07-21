#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
use candle_core::{CpuStorage, Layout, Shape};
#[allow(unused_imports)]
use candle_core::{DType, Device, Tensor, Var};
use std::ops::Deref;

pub struct BackwardAntiAliasSilhouette {
    edge2vtx_contour: Tensor,
    transform_world2pix: Tensor,
    pix2tri: Tensor,
}

impl candle_core::InplaceOp3 for BackwardAntiAliasSilhouette {
    fn name(&self) -> &'static str {
        "BackwardAntiAliasSilhouette"
    }
    fn cpu_fwd(
        &self,
        dldw_vtx2xyz: &mut CpuStorage,
        _l_dldw_vtx2xyz: &Layout,
        vtx2xyz: &CpuStorage,
        _l_vtx2xyz: &Layout,
        dldw_pix2occl: &CpuStorage,
        l_dldw_pix2occul: &Layout,
    ) -> candle_core::Result<()> {
        assert!(self.pix2tri.device().is_cpu());
        assert!(self.transform_world2pix.device().is_cpu());
        assert!(self.edge2vtx_contour.device().is_cpu());
        let dldw_vtx2xyz = match dldw_vtx2xyz {
            CpuStorage::F32(cpu_storage) => cpu_storage,
            _ => panic!(),
        };
        let dldw_pix2occl = dldw_pix2occl.as_slice::<f32>()?;
        get_cpu_slice_and_storage_from_tensor!(
            edge2vtx_contour,
            _s_edge2vtx_contour,
            self.edge2vtx_contour,
            u32
        );
        let img_shape = (l_dldw_pix2occul.dim(1)?, l_dldw_pix2occul.dim(0)?);
        get_cpu_slice_and_storage_from_tensor!(
            transform_world2pix,
            _s_transform_world2pix,
            self.transform_world2pix,
            f32
        );
        let transform_world2pix = arrayref::array_ref![transform_world2pix, 0, 16];
        get_cpu_slice_and_storage_from_tensor!(pix2tri, _s_pix2tri, self.pix2tri, u32);
        del_msh_cpu::silhouette::backward_wrt_vtx2xyz(
            edge2vtx_contour,
            vtx2xyz.as_slice()?,
            dldw_vtx2xyz,
            transform_world2pix,
            img_shape,
            dldw_pix2occl,
            pix2tri,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        dldw_vtx2xyz: &mut CudaStorage,
        _l_dldw_vtx2xyz: &Layout,
        vtx2xyz: &CudaStorage,
        _l_vtx2xyz: &Layout,
        dldw_pix2occul: &CudaStorage,
        l_dldw_pix2occul: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        // let cuda_device = dldw_vtx2xyz.device();
        assert!(self.edge2vtx_contour.device().is_cuda());
        assert!(self.transform_world2pix.device().is_cuda());
        assert!(self.pix2tri.device().is_cuda());
        get_cuda_slice_device_from_storage_f32!(vtx2xyz, _dev_vtx2xyy, vtx2xyz);
        get_cuda_slice_device_from_storage_f32!(dldw_pix2occl, _dev_dldw_pix2occl, dldw_pix2occul);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            edge2vtx_contour,
            s_edge2vtx_contour,
            _l_edge2vtx_contour,
            self.edge2vtx_contour,
            u32
        );
        let img_shape = (l_dldw_pix2occul.dim(1)?, l_dldw_pix2occul.dim(0)?);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            transform_world2pix,
            s_transform_world2pix,
            _l_transform_world2pix,
            self.transform_world2pix,
            f32
        );
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            pix2tri,
            s_pix2tri,
            _l_pix2tri,
            self.pix2tri,
            u32
        );
        get_cuda_slice_device_from_storage_f32!(dldw_vtx2xyz, dev_dldw_vtx2xyz, dldw_vtx2xyz);
        del_msh_cudarc::silhouette::backward_wrt_vtx2xyz(
            &dev_dldw_vtx2xyz.cuda_stream(),
            edge2vtx_contour,
            vtx2xyz,
            &mut dldw_vtx2xyz.slice_mut(..),
            transform_world2pix,
            img_shape,
            dldw_pix2occl,
            pix2tri,
        )
        .w()?;
        Ok(())
    }
}

pub struct AntiAliasSilhouette {
    pix2tri: Tensor,
    edge2vtx_contour: Tensor,
    transform_world2pix: Tensor,
}

impl candle_core::CustomOp1 for AntiAliasSilhouette {
    fn name(&self) -> &'static str {
        "anti_alias_silhouette"
    }

    fn cpu_fwd(
        &self,
        vtx2xyz: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        assert_eq!(l_vtx2xyz.dim(1)?, 3);
        let img_shape = (self.pix2tri.dim(1)?, self.pix2tri.dim(0)?);
        get_cpu_slice_and_storage_from_tensor!(
            edge2vtx_contour,
            _s_edge2vtx_contour,
            self.edge2vtx_contour,
            u32
        );
        get_cpu_slice_and_storage_from_tensor!(pix2tri, _s_pix2tri, self.pix2tri, u32);
        get_cpu_slice_and_storage_from_tensor!(
            transform_world2pix,
            _s_transform_world2pix,
            self.transform_world2pix,
            f32
        );
        let transform_world2pix = arrayref::array_ref![transform_world2pix, 0, 16];
        let vtx2xyz = vtx2xyz.as_slice()?;
        let mut img = vec![0f32; img_shape.0 * img_shape.1];
        {
            // initial aliased silhouette image
            use rayon::prelude::*;
            img.par_iter_mut()
                .zip(pix2tri.par_iter())
                .for_each(|(a, &b)| {
                    if b != u32::MAX {
                        *a = 1f32;
                    }
                });
        }
        del_msh_cpu::silhouette::update_image(
            edge2vtx_contour,
            vtx2xyz,
            transform_world2pix,
            img_shape,
            &mut img,
            pix2tri,
        );
        Ok((CpuStorage::F32(img), (img_shape.1, img_shape.0).into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        vtx2xyz: &CudaStorage,
        _l_vtx2xyz: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::WrapErr;
        let device = &vtx2xyz.device;
        let img_shape = (self.pix2tri.dim(1)?, self.pix2tri.dim(0)?);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            pix2tri,
            s_pix2tri,
            _l_pix2tril,
            self.pix2tri,
            u32
        );
        assert_eq!(self.edge2vtx_contour.dim(1)?, 2);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            edge2vtx_contour,
            s_edge2vtx_contour,
            _l_edge2vtx_contour,
            self.edge2vtx_contour,
            u32
        );
        assert_eq!(self.transform_world2pix.dims(), &[16]);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            transform_world2pix,
            s_transform_world2pix,
            _l_transform_world2pix,
            self.transform_world2pix,
            f32
        );
        let vtx2xyz = vtx2xyz.as_cuda_slice()?;
        //let img = candle_core::cuda_backend::cua
        let mut pix2occu = del_msh_cudarc::silhouette::compute_with_alias(
            &device.cuda_stream(),
            img_shape,
            pix2tri,
        )
        .w()?;
        del_msh_cudarc::silhouette::remove_alias(
            &device.cuda_stream(),
            edge2vtx_contour,
            img_shape,
            &mut pix2occu,
            pix2tri,
            vtx2xyz,
            transform_world2pix,
        )
        .w()?;
        let s_pix2occu = candle_core::CudaStorage::wrap_cuda_slice(pix2occu, device.clone());
        Ok((s_pix2occu, (img_shape.1, img_shape.0).into()))
    }

    fn bwd(
        &self,
        vtx2xyz: &Tensor,
        pix2occl: &Tensor,
        dldw_pix2occl: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        assert_eq!(pix2occl.shape(), dldw_pix2occl.shape());
        let dldw_vtx2xyz = Tensor::zeros_like(vtx2xyz)?;
        let op = BackwardAntiAliasSilhouette {
            edge2vtx_contour: self.edge2vtx_contour.clone(),
            transform_world2pix: self.transform_world2pix.clone(),
            pix2tri: self.pix2tri.clone(),
        };
        dldw_vtx2xyz.inplace_op3(vtx2xyz, dldw_pix2occl, &op)?;
        Ok(Some(dldw_vtx2xyz))
    }
}

#[test]
fn test_cpu() -> anyhow::Result<()> {
    let (tri2vtx, vtx2xyz, vtx2idx, idx2vtx, edge2vtx, edge2tri) = {
        let (tri2vtx, vtx2xyz) =
            del_msh_cpu::trimesh3_primitive::sphere_yup::<u32, f32>(0.5, 64, 64);
        let num_vtx = vtx2xyz.len() / 3;
        let (vtx2idx, idx2vtx) =
            del_msh_cpu::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx, false);
        let edge2vtx = del_msh_cpu::edge2vtx::from_triangle_mesh(&tri2vtx, num_vtx);
        let num_tri = tri2vtx.len() / 3;
        let num_edge = edge2vtx.len() / 2;
        let edge2tri =
            del_msh_cpu::edge2elem::from_edge2vtx_of_tri2vtx(&edge2vtx, &tri2vtx, num_vtx);
        //
        let vtx2idx = Tensor::from_vec(vtx2idx, num_vtx + 1, &Device::Cpu)?;
        let num_idx = idx2vtx.len();
        let idx2vtx = Tensor::from_vec(idx2vtx, num_idx, &Device::Cpu)?;
        let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &Device::Cpu)?;
        let vtx2xyz = Var::from_vec(vtx2xyz, (num_vtx, 3), &Device::Cpu)?;
        let edge2vtx = Tensor::from_vec(edge2vtx, (num_edge, 2), &Device::Cpu)?;
        let edge2tri = Tensor::from_vec(edge2tri, (num_edge, 2), &Device::Cpu)?;
        (tri2vtx, vtx2xyz, vtx2idx, idx2vtx, edge2vtx, edge2tri)
    };
    let bvhdata = crate::bvhnode2aabb::BvhForTriMesh::new(
        tri2vtx.dims2()?.0,
        vtx2xyz.dims2()?.1,
        &Device::Cpu,
    )?;
    let img_asp = 1.5;
    let img_shape = (((16 * 6) as f32 * img_asp) as usize, 16 * 6);
    let cam_projection =
        del_geo_core::mat4_col_major::camera_perspective_blender(img_asp, 24f32, 0.3, 10.0, true);
    let cam_modelview =
        del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 2.0], 0., 0., 0.);
    // ----------------------
    let transform_world2ndc =
        del_geo_core::mat4_col_major::mult_mat_col_major(&cam_projection, &cam_modelview);
    let transform_ndc2world =
        del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap();
    let transform_world2pix = {
        let transform_ndc2pix = del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape);
        let transform_ndc2pix =
            del_geo_core::mat4_col_major::from_mat3_col_major_adding_z(&transform_ndc2pix);
        del_geo_core::mat4_col_major::mult_mat_col_major(&transform_ndc2pix, &transform_world2ndc)
    };
    let transform_ndc2world = Tensor::from_vec(transform_ndc2world.to_vec(), 16, &Device::Cpu)?;
    let transform_world2pix = Tensor::from_vec(transform_world2pix.to_vec(), 16, &Device::Cpu)?;
    let transform_world2ndc = Tensor::from_vec(transform_world2ndc.to_vec(), 16, &Device::Cpu)?;
    let img_trg = {
        let transform_ndc2pix = del_geo_core::mat3_col_major::from_transform_ndc2pix(img_shape);
        let mut img_trg = vec![0f32; img_shape.0 * img_shape.1];
        del_canvas::rasterize::circle2::fill::<f32, f32>(
            &mut img_trg,
            img_shape.0,
            &[0.0, 0.0],
            &transform_ndc2pix,
            (img_shape.1 as f32) * 0.4f32,
            1f32,
        );
        del_canvas::write_png_from_float_image_grayscale(
            "../target/silhouette_trg.png",
            img_shape,
            &img_trg,
        )?;

        Tensor::from_vec(img_trg, (img_shape.1, img_shape.0), &Device::Cpu)?
    };
    // ---------------------------------------------------
    for iter in 0..300 {
        bvhdata.compute(&tri2vtx, &vtx2xyz)?;
        let pix2tri = crate::pix2tri::from_trimesh3(
            &tri2vtx,
            &vtx2xyz,
            &bvhdata.bvhnodes,
            &bvhdata.bvhnode2aabb,
            img_shape,
            &transform_ndc2world,
        )?;
        let edge2vtx_contour = {
            // extract edges on the contour
            let layer_contour = crate::edge2vtx_trimesh3::Layer {
                tri2vtx: tri2vtx.clone(),
                edge2vtx: edge2vtx.clone(),
                edge2tri: edge2tri.clone(),
                transform_world2ndc: transform_world2ndc.clone(),
            };
            vtx2xyz.apply_op1(layer_contour)?
        };
        let img = {
            let layer_silhouette = AntiAliasSilhouette {
                edge2vtx_contour: edge2vtx_contour.clone(),
                pix2tri: pix2tri.clone(),
                transform_world2pix: transform_world2pix.clone(),
            };
            vtx2xyz.apply_op1(layer_silhouette)?
        };
        if iter % 10 == 0 {
            del_canvas::write_png_from_float_image_grayscale(
                format!("../target/del-raycast-candle__silhouette_{iter}.png"),
                img_shape,
                &img.flatten_all()?.to_vec1::<f32>()?,
            )?;
            {
                let vtx2xyz = vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
                let tri2vtx = tri2vtx.flatten_all()?.to_vec1::<u32>()?;
                del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
                    format!("../target/del-raycast-candle__silhouette_{iter}.obj"),
                    &tri2vtx,
                    &vtx2xyz,
                    3,
                )?;
            }
        }
        let loss = img.sub(&img_trg)?.sqr()?.sum_all()?;
        println!("loss: {}", loss.to_vec0::<f32>()?);
        let grads = loss.backward()?;
        #[cfg(feature = "cuda")]
        if iter == 0 {
            let dldw_vtx2xyz_cpu = grads
                .get(&vtx2xyz)
                .unwrap()
                .flatten_all()?
                .to_vec1::<f32>()?;
            dbg!("hoge");
            let device = Device::cuda_if_available(0)?;
            let tri2vtx = tri2vtx.to_device(&device)?;
            let vtx2xyz = Var::from_tensor(&vtx2xyz.to_device(&device)?)?;
            let edge2vtx = edge2vtx.to_device(&device)?;
            let edge2tri = edge2tri.to_device(&device)?;
            let img_trg = img_trg.to_device(&device)?;
            let transform_ndc2world = transform_ndc2world.to_device(&device)?;
            let transform_world2ndc = transform_world2ndc.to_device(&device)?;
            let transform_world2pix = transform_world2pix.to_device(&device)?;
            let bvhdata = crate::bvhnode2aabb::BvhForTriMesh::from_trimesh(&tri2vtx, &vtx2xyz)?;
            let pix2tri_cpu = pix2tri.flatten_all()?.to_vec1::<u32>()?;
            let pix2tri = Tensor::zeros((img_shape.1, img_shape.0), DType::U32, &device)?;
            let layer = crate::pix2tri::Pix2Tri {
                bvhnodes: bvhdata.bvhnodes,
                bvhnode2aabb: bvhdata.bvhnode2aabb,
                transform_ndc2world,
            };
            pix2tri.inplace_op3(&tri2vtx, &vtx2xyz, &layer)?;
            let pix2tri_gpu = pix2tri.flatten_all()?.to_vec1::<u32>()?;
            pix2tri_cpu
                .iter()
                .zip(pix2tri_gpu.iter())
                .for_each(|(&a, &b)| {
                    assert_eq!(a, b);
                    // println!("{} {}", a, b);
                });
            let edge2vtx_contour_cpu = edge2vtx_contour.flatten_all()?.to_vec1::<u32>()?;
            let layer_contour = crate::edge2vtx_trimesh3::Layer {
                tri2vtx: tri2vtx.clone(),
                edge2vtx: edge2vtx.clone(),
                edge2tri: edge2tri.clone(),
                transform_world2ndc: transform_world2ndc.clone(),
            };
            let edge2vtx_contour = vtx2xyz.apply_op1(layer_contour)?;
            let edge2vtx_contour_gpu = edge2vtx_contour.flatten_all()?.to_vec1::<u32>()?;
            edge2vtx_contour_cpu
                .iter()
                .zip(edge2vtx_contour_gpu.iter())
                .for_each(|(&a, &b)| {
                    assert_eq!(a, b);
                    // println!("{} {}", a, b);
                });
            let img_cpu = img.flatten_all()?.to_vec1::<f32>()?;
            let layer_silhouette = AntiAliasSilhouette {
                edge2vtx_contour: edge2vtx_contour.clone(),
                pix2tri: pix2tri.clone(),
                transform_world2pix: transform_world2pix.clone(),
            };
            let img = vtx2xyz.apply_op1(layer_silhouette)?;
            let img_gpu = img.flatten_all()?.to_vec1::<f32>()?;
            del_canvas::write_png_from_float_image_grayscale(
                "../target/del-raycast-candle__silhouette_cuda.png",
                img_shape,
                &img_gpu,
            )?;
            img_cpu.iter().zip(img_gpu.iter()).for_each(|(&a, &b)| {
                assert!((a - b).abs() < 1.0e-5, "{} {}", a, b);
                // println!("{} {}", a, b);
            });
            let loss = img.sub(&img_trg)?.sqr()?.sum_all()?;
            println!("loss: {}", loss.to_vec0::<f32>()?);
            let grads = loss.backward()?;
            let dldw_vtx2xyz = grads.get(&vtx2xyz).unwrap();
            let dldw_vtx2xyz_gpu = dldw_vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
            dldw_vtx2xyz_cpu
                .iter()
                .zip(dldw_vtx2xyz_gpu.iter())
                .for_each(|(&a, &b)| {
                    assert!((a - b).abs() < 1.0e-3, "{} {}", a, b);
                    //println!("{} {}", a, b);
                });
        }
        {
            let layer = crate::laplacian_smoothing::LaplacianSmoothing {
                vtx2idx: vtx2idx.clone(),
                idx2vtx: idx2vtx.clone(),
                lambda: 30.0,
                num_iter: 200,
            };
            let dldw_vtx2xyz = grads.get(&vtx2xyz).unwrap();
            let grad = dldw_vtx2xyz.apply_op1_no_bwd(&layer)?;
            let grad = (grad * 0.003)?;
            vtx2xyz.set(&vtx2xyz.sub(&(grad))?)?;
        }
    }
    Ok(())
}
