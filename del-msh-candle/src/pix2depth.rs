#[cfg(feature = "cuda")]
use candle_core::CudaStorage;
use candle_core::{CpuStorage, Layout, Shape, Tensor};
use std::ops::Deref;

pub fn update_bwd_wrt_vtx2xyz(
    dw_vtx2xyz: &mut [f32],
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
    (width, height): (usize, usize),
    pix2tri: &[u32],
    dw_pix2depth: &[f32],
    transform_ndc2world: &[f32; 16],
) {
    let transform_world2ndc =
        del_geo_core::mat4_col_major::try_inverse(transform_ndc2world).unwrap();
    for i_pix in 0..width * height {
        let i_tri = pix2tri[i_pix];
        if i_tri == u32::MAX {
            continue;
        }
        let (ray_org, ray_dir) =
            del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinate(
                (i_pix % width, i_pix / width),
                &(width, height),
                transform_ndc2world,
            );
        let i_tri = i_tri as usize;
        let (p0, p1, p2) = del_msh_cpu::trimesh3::to_corner_points(tri2vtx, vtx2xyz, i_tri);
        let dw_depth = {
            let t =
                del_geo_core::tri3::intersection_against_line(&p0, &p1, &p2, &ray_org, &ray_dir)
                    .unwrap();
            let pos = del_geo_core::vec3::axpy(t, &ray_dir, &ray_org);
            let jacb = del_geo_core::mat4_col_major::jacobian_transform(&transform_world2ndc, &pos);
            let tmp = del_geo_core::mat3_col_major::mult_vec(&jacb, &ray_dir);
            tmp[2] * 0.5 * dw_pix2depth[i_pix]
        };
        let (_t, _u, _v, dw_p0, dw_p1, dw_p2) =
            del_geo_core::tri3::intersection_against_line_bwd_wrt_tri(
                &p0, &p1, &p2, &ray_org, &ray_dir, dw_depth, 0., 0.,
            );
        use del_geo_core::vec3::Vec3;
        let iv0 = tri2vtx[i_tri * 3] as usize;
        let iv1 = tri2vtx[i_tri * 3 + 1] as usize;
        let iv2 = tri2vtx[i_tri * 3 + 2] as usize;
        arrayref::array_mut_ref![dw_vtx2xyz, iv0 * 3, 3].add_in_place(&dw_p0);
        arrayref::array_mut_ref![dw_vtx2xyz, iv1 * 3, 3].add_in_place(&dw_p1);
        arrayref::array_mut_ref![dw_vtx2xyz, iv2 * 3, 3].add_in_place(&dw_p2);
    }
}

struct BackwardPix2Depth {
    dw_pix2depth: Tensor,
    pix2tri: Tensor,
    transform_ndc2world: Tensor,
}

impl candle_core::InplaceOp3 for BackwardPix2Depth {
    fn name(&self) -> &'static str {
        "bwd_pix2depth_wrt_vtx2xyz"
    }
    fn cpu_fwd(
        &self,
        dw_vtx2xyz: &mut CpuStorage,
        _l_dw_vtx2xyz: &Layout,
        tri2vtx: &CpuStorage,
        _l_tri2vtx: &Layout,
        vtx2xyz: &CpuStorage,
        _l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        let img_shape = (self.pix2tri.dim(1)?, self.pix2tri.dim(0)?);
        let dw_vtx2xyz = match dw_vtx2xyz {
            CpuStorage::F32(cpu_storage) => cpu_storage,
            _ => panic!(),
        };
        get_cpu_slice_and_storage_from_tensor!(pix2tri, storage, self.pix2tri, u32);
        get_cpu_slice_and_storage_from_tensor!(dw_pix2depth, storage, self.dw_pix2depth, f32);
        get_cpu_slice_and_storage_from_tensor!(
            transform_ndc2world,
            storage,
            self.transform_ndc2world,
            f32
        );
        let transform_ndc2world = arrayref::array_ref![transform_ndc2world, 0, 16];
        update_bwd_wrt_vtx2xyz(
            dw_vtx2xyz.as_mut_slice(),
            tri2vtx.as_slice::<u32>()?,
            vtx2xyz.as_slice::<f32>()?,
            img_shape,
            pix2tri,
            dw_pix2depth,
            transform_ndc2world,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        dw_vtx2xyz: &mut CudaStorage,
        l_dw_vtx2xyz: &Layout,
        tri2vtx: &CudaStorage,
        l_tri2vtx: &Layout,
        vtx2xyz: &CudaStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::CudaStorageSlice;
        use candle_core::cuda_backend::WrapErr;
        assert_eq!(l_dw_vtx2xyz.shape().dims2()?, l_vtx2xyz.shape().dims2()?);
        assert_eq!(l_tri2vtx.shape().dims2()?.1, 3);
        let img_shape = (self.pix2tri.dim(1)?, self.pix2tri.dim(0)?);
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            pix2tri,
            storage,
            _layout,
            self.pix2tri,
            u32
        );
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            dw_pix2depth,
            storage,
            _layout,
            self.dw_pix2depth,
            f32
        );
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            transform_ndc2world,
            storage,
            _layout,
            self.transform_ndc2world,
            f32
        );
        get_cuda_slice_and_device_from_storage_f32!(dw_vtx2xyz, device_dw_vtx2xyz, dw_vtx2xyz);
        get_cuda_slice_and_device_from_storage_u32!(tri2vtx, device_tri2vtx, tri2vtx);
        get_cuda_slice_and_device_from_storage_f32!(vtx2xyz, device_vtx2xyz, vtx2xyz);
        assert!(device_dw_vtx2xyz.same_device(device_tri2vtx));
        assert!(device_dw_vtx2xyz.same_device(device_vtx2xyz));
        del_raycast_cudarc::pix2depth::bwd_wrt_vtx2xyz(
            device_dw_vtx2xyz,
            img_shape,
            &mut dw_vtx2xyz.slice_mut(..),
            pix2tri,
            tri2vtx,
            vtx2xyz,
            dw_pix2depth,
            transform_ndc2world,
        )
        .w()?;
        Ok(())
    }
}

// ---------------------------------

pub struct Pix2Depth {
    pub tri2vtx: Tensor,
    pub pix2tri: Tensor,
    pub transform_ndc2world: Tensor, // transform column major
}

impl candle_core::CustomOp1 for Pix2Depth {
    fn name(&self) -> &'static str {
        "pix2depth"
    }

    fn cpu_fwd(
        &self,
        vtx2xyz: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (_num_vtx, three) = l_vtx2xyz.shape().dims2()?;
        assert_eq!(three, 3);
        let vtx2xyz = vtx2xyz.as_slice::<f32>()?;
        get_cpu_slice_and_storage_from_tensor!(tri2vtx, storage, self.tri2vtx, u32);
        get_cpu_slice_and_storage_from_tensor!(pix2tri, storage, self.pix2tri, u32);
        get_cpu_slice_and_storage_from_tensor!(
            transform_ndc2world,
            storage,
            self.transform_ndc2world,
            f32
        );
        let transform_ndc2world = arrayref::array_ref![transform_ndc2world, 0, 16];
        let transform_world2ndc =
            del_geo_core::mat4_col_major::try_inverse(transform_ndc2world).unwrap();
        //
        let img_shape = (self.pix2tri.dim(1)?, self.pix2tri.dim(0)?);
        let fn_pix2depth = |i_pix: usize| -> Option<f32> {
            let (i_w, i_h) = (i_pix % img_shape.0, i_pix / img_shape.0);
            let i_tri = pix2tri[i_h * img_shape.0 + i_w];
            if i_tri == u32::MAX {
                return None;
            }
            let (ray_org, ray_dir) =
                del_geo_core::mat4_col_major::ray_from_transform_ndc2world_and_pixel_coordinate(
                    (i_w, i_h),
                    &img_shape,
                    transform_ndc2world,
                );
            let tri = del_msh_cpu::trimesh3::to_tri3(tri2vtx, vtx2xyz, i_tri as usize);
            let coeff = del_geo_core::tri3::intersection_against_line(
                tri.p0, tri.p1, tri.p2, &ray_org, &ray_dir,
            )
            .unwrap();
            let pos_world = del_geo_core::vec3::axpy(coeff, &ray_dir, &ray_org);
            let pos_ndc = del_geo_core::mat4_col_major::transform_homogeneous(
                &transform_world2ndc,
                &pos_world,
            )
            .unwrap();
            let depth_ndc = (pos_ndc[2] + 1f32) * 0.5f32;
            Some(depth_ndc)
        };
        let mut pix2depth = vec![0f32; img_shape.0 * img_shape.1];
        use rayon::prelude::*;
        pix2depth
            .par_iter_mut()
            .enumerate()
            .for_each(|(i_pix, depth)| {
                *depth = fn_pix2depth(i_pix).unwrap_or(0f32);
            });
        let shape = candle_core::Shape::from((img_shape.1, img_shape.0));
        let storage = candle_core::WithDType::to_cpu_storage_owned(pix2depth);
        Ok((storage, shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        vtx2xyz: &CudaStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        use candle_core::backend::BackendDevice;
        use candle_core::cuda_backend::CudaStorage;
        use candle_core::cuda_backend::WrapErr;
        assert_eq!(l_vtx2xyz.dim(1)?, 3);
        let img_shape = (self.pix2tri.dim(1)?, self.pix2tri.dim(0)?);
        //get_cuda_slice_from_tensor!(vtx2xyz, device_vtx2xyz, vtx2xyz);
        let device = &vtx2xyz.device;
        assert!(device.same_device(self.pix2tri.device().as_cuda_device()?));
        assert!(device.same_device(self.tri2vtx.device().as_cuda_device()?));
        assert!(device.same_device(self.transform_ndc2world.device().as_cuda_device()?));
        let vtx2xyz = vtx2xyz.as_cuda_slice::<f32>()?;
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            pix2tri,
            storage_pix2tri,
            _layout_pix2tri,
            self.pix2tri,
            u32
        );
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            tri2vtx,
            storage_tri2vtx,
            _layout_tri2vtx,
            self.tri2vtx,
            u32
        );
        get_cuda_slice_and_storage_and_layout_from_tensor!(
            transform_ndc2world,
            storage_transform_ndc2world,
            _layout_transform_ndc2world,
            self.transform_ndc2world,
            f32
        );
        // let mut pix2depth = unsafe { device.alloc::<f32>(img_shape.0 * img_shape.1) }.w()?;
        let mut pix2depth = device.alloc_zeros::<f32>(img_shape.0 * img_shape.1).w()?;
        del_raycast_cudarc::pix2depth::fwd(
            device,
            img_shape,
            &mut pix2depth,
            pix2tri,
            tri2vtx,
            vtx2xyz,
            transform_ndc2world,
        )
        .w()?;
        let pix2depth = CudaStorage::wrap_cuda_slice(pix2depth, device.clone());
        Ok((pix2depth, (img_shape.1, img_shape.0).into()))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(
        &self,
        vtx2xyz: &Tensor,
        _pix2depth: &Tensor,
        dw_pix2depth: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let dw_vtx2xyz = Tensor::zeros_like(vtx2xyz)?;
        let op = BackwardPix2Depth {
            pix2tri: self.pix2tri.clone(),
            transform_ndc2world: self.transform_ndc2world.clone(),
            dw_pix2depth: dw_pix2depth.clone(),
        };
        dw_vtx2xyz.inplace_op3(&self.tri2vtx, vtx2xyz, &op)?;
        Ok(Some(dw_vtx2xyz))
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use candle_core::CudaStorage;
    use candle_core::{Device, Tensor};

    fn render(
        device: &Device,
        tri2vtx: &Tensor,
        vtx2xyz: &Tensor,
        img_shape: (usize, usize),
        transform_ndc2world: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let tri2vtx = tri2vtx.to_device(&device)?;
        let vtx2xyz = vtx2xyz.to_device(&device)?;
        let transform_ndc2world = transform_ndc2world.to_device(&device)?;
        let bvhdata = crate::bvhnode2aabb::BvhForTriMesh::from_trimesh(&tri2vtx, &vtx2xyz)?;
        let pix2tri = crate::pix2tri::from_trimesh3(
            &tri2vtx,
            &vtx2xyz,
            &bvhdata.bvhnodes,
            &bvhdata.bvhnode2aabb,
            img_shape,
            &transform_ndc2world,
        )?;
        /*
        {
            // test pix2tri
            let a = pix2tri.flatten_all()?.to_vec1::<u32>()?;
            let a = a.iter().map(|&v| if v == u32::MAX { 0f32 } else { 1f32} ).collect::<Vec<_>>();
            del_canvas::write_png_from_float_image_grayscale(
                "../target/pix2depth_pix2tri_test.png",
                img_shape,
                &a,
            ).unwrap();
            dbg!(img_shape);
        }
         */
        println!("render depth start: {:?}", device);
        let render = crate::pix2depth::Pix2Depth {
            tri2vtx: tri2vtx.clone(),
            pix2tri: pix2tri.clone(),
            transform_ndc2world: transform_ndc2world.clone(),
        };
        let a = Ok(vtx2xyz.apply_op1(render)?);
        println!("render depth end: {:?}", device);
        a
    }

    #[test]
    fn test_optimize_depth() -> anyhow::Result<()> {
        let (tri2vtx, vtx2xyz) =
            del_msh_cpu::trimesh3_primitive::sphere_yup::<u32, f32>(0.8, 32, 32);
        let vtx2xyz = {
            let mut vtx2xyz_new = vtx2xyz.clone();
            del_msh_cpu::vtx2xyz::translate_then_scale(
                &mut vtx2xyz_new,
                &vtx2xyz,
                &[0.2, 0.0, 0.0],
                1.0,
            );
            vtx2xyz_new
        };
        let num_vtx = vtx2xyz.len() / 3;
        let (vtx2idx, idx2vtx) =
            del_msh_cpu::vtx2vtx::from_uniform_mesh(&tri2vtx, 3, num_vtx, false);
        let num_tri = tri2vtx.len() / 3;
        let tri2vtx = Tensor::from_vec(tri2vtx, (num_tri, 3), &Device::Cpu)?;
        let vtx2xyz = candle_core::Var::from_vec(vtx2xyz, (num_vtx, 3), &Device::Cpu)?;
        let vtx2idx = Tensor::from_vec(vtx2idx, num_vtx + 1, &Device::Cpu)?;
        let num_idx = idx2vtx.len();
        let idx2vtx = Tensor::from_vec(idx2vtx, num_idx, &Device::Cpu)?;
        //
        let img_shape = (400, 300);
        // let transform_ndc2world = del_geo_core::mat4_col_major::from_identity::<f32>();
        let transform_ndc2world = {
            let img_asp = (img_shape.0 as f32) / (img_shape.1 as f32);
            let cam_projection = del_geo_core::mat4_col_major::camera_perspective_blender(
                img_asp, 35f32, 2.0, 5.0, true,
            );
            let cam_modelview =
                del_geo_core::mat4_col_major::camera_external_blender(&[0., 0., 3.0], 0., 0., 0.);
            let transform_world2ndc =
                del_geo_core::mat4_col_major::mult_mat_col_major(&cam_projection, &cam_modelview);
            del_geo_core::mat4_col_major::try_inverse(&transform_world2ndc).unwrap()
        };
        let (pix2depth_trg, pix2mask) = {
            let mut img2depth_trg = vec![0f32; img_shape.0 * img_shape.1];
            let mut img2mask = vec![0f32; img_shape.0 * img_shape.1];
            for i_h in 0..img_shape.1 {
                for i_w in 0..img_shape.0 {
                    /*
                    let (ray_org, ray_dir)
                        = del_raycast_core::cam3::ray3_homogeneous(
                        (i_w, i_h), img_shape, &transform_ndc2world);
                     */
                    let x = 2f32 * (i_w as f32) / (img_shape.0 as f32) - 1f32;
                    let y = 1f32 - 2f32 * (i_h as f32) / (img_shape.1 as f32);
                    let r = (x * x + y * y).sqrt();
                    if r > 0.3 {
                        continue;
                    }
                    img2depth_trg[i_h * img_shape.0 + i_w] = 0.6;
                    img2mask[i_h * img_shape.0 + i_w] = 1.0;
                }
            }
            let img2depth_trg =
                Tensor::from_vec(img2depth_trg, (img_shape.1, img_shape.0), &Device::Cpu)?;
            let img2mask = Tensor::from_vec(img2mask, (img_shape.1, img_shape.0), &Device::Cpu)?;
            (img2depth_trg, img2mask)
        };
        let transform_ndc2world = Tensor::from_vec(transform_ndc2world.to_vec(), 16, &Device::Cpu)?;
        {
            // output target images
            let pix2depth_trg = pix2depth_trg.flatten_all()?.to_vec1::<f32>()?;
            del_canvas::write_png_from_float_image_grayscale(
                "../target/pix2depth_trg.png",
                img_shape,
                &pix2depth_trg,
            )?;
            //
            let pix2mask = pix2mask.flatten_all()?.to_vec1::<f32>()?;
            del_canvas::write_png_from_float_image_grayscale(
                "../target/pix2mask.png",
                img_shape,
                &pix2mask,
            )?;
        }
        #[cfg(feature = "cuda")]
        {
            let conj = Tensor::rand(0f32, 1f32, (img_shape.1, img_shape.0), &Device::Cpu)?;
            // try gpu depth render
            let pix2depth_cpu = render(
                &Device::Cpu,
                &tri2vtx,
                &vtx2xyz,
                img_shape,
                &transform_ndc2world,
            )?;
            {
                let pix2depth_cpu = pix2depth_cpu.flatten_all()?.to_vec1::<f32>()?;
                del_canvas::write_png_from_float_image_grayscale(
                    "../target/pix2depth_cpu.png",
                    img_shape,
                    &pix2depth_cpu,
                )?;
            }
            let loss_cpu = pix2depth_cpu.mul(&conj)?.sum_all()?;
            let grad_vtx2xyz_cpu = loss_cpu.backward()?.get(&vtx2xyz).unwrap().to_owned();
            let loss_cpu = loss_cpu.to_vec0::<f32>()?;
            let grad_vtx2xyz_cpu = grad_vtx2xyz_cpu.flatten_all()?.to_vec1::<f32>()?;
            let pix2depth_cpu = pix2depth_cpu.flatten_all()?.to_vec1::<f32>()?;
            //
            let device = Device::new_cuda(0)?;
            let conj_cuda = conj.to_device(&device)?;
            let pix2depth_cuda =
                render(&device, &tri2vtx, &vtx2xyz, img_shape, &transform_ndc2world)?;
            let loss_cuda = pix2depth_cuda.mul(&conj_cuda)?.sum_all()?;
            let grad_vtx2xyz_cuda = loss_cuda.backward()?.get(&vtx2xyz).unwrap().to_owned();
            let loss_cuda = loss_cuda.to_vec0::<f32>()?;
            let grad_vtx2xyz_cuda = grad_vtx2xyz_cuda.flatten_all()?.to_vec1::<f32>()?;
            let pix2depth_cuda = pix2depth_cuda.flatten_all()?.to_vec1::<f32>()?;
            assert!(
                (loss_cpu - loss_cuda).abs() < 1.0e-1,
                "{} {} {}",
                loss_cuda,
                loss_cpu,
                loss_cuda - loss_cpu
            );
            pix2depth_cpu
                .iter()
                .zip(pix2depth_cuda.iter())
                .for_each(|(a, b)| {
                    assert!((a - b).abs() < 1.0e-6);
                });
            grad_vtx2xyz_cpu
                .iter()
                .enumerate()
                .zip(grad_vtx2xyz_cuda.iter())
                .for_each(|((i, a), b)| {
                    assert!(
                        (a - b).abs() < 1.0e-3,
                        "{} {} {} {}",
                        i,
                        a,
                        b,
                        (a - b).abs()
                    );
                    //println!("{} {} {} {}", i, a, b, (a - b).abs());
                });
        }

        /*
        let mut optimizer = crate::gd_with_laplacian_reparam::Optimizer::new(
            vtx2xyz.clone(),
            0.001,
            tri2vtx.clone(),
            vtx2xyz.dims2()?.0,
            0.8,
        )?;
         */

        // let mut optimizer = candle_nn::AdamW::new_lr(vec!(vtx2xyz.clone()), 0.01)?;

        for itr in 0..100 {
            let bvhdata = crate::bvhnode2aabb::BvhForTriMesh::from_trimesh(&tri2vtx, &vtx2xyz)?;
            let pix2tri = crate::pix2tri::from_trimesh3(
                &tri2vtx,
                &vtx2xyz,
                &bvhdata.bvhnodes,
                &bvhdata.bvhnode2aabb,
                img_shape,
                &transform_ndc2world,
            )?;
            let render = crate::pix2depth::Pix2Depth {
                tri2vtx: tri2vtx.clone(),
                pix2tri: pix2tri.clone(),
                transform_ndc2world: transform_ndc2world.clone(),
            };
            let pix2depth = vtx2xyz.apply_op1(render)?;
            let pix2diff = pix2depth.sub(&pix2depth_trg)?.mul(&pix2mask)?;
            {
                let pix2depth = pix2depth.flatten_all()?.to_vec1::<f32>()?;
                del_canvas::write_png_from_float_image_grayscale(
                    "../target/pix2depth.png",
                    img_shape,
                    &pix2depth,
                )?;
                let pix2diff = (pix2diff.clone() * 10.0)?
                    .abs()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                del_canvas::write_png_from_float_image_grayscale(
                    "../target/pix2diff.png",
                    img_shape,
                    &pix2diff,
                )?;
            }
            let loss = pix2diff.sqr()?.sum_all()?;
            println!("loss: {}", loss.to_vec0::<f32>()?);
            {
                let grads = loss.backward()?;
                let grad = grads.get(&vtx2xyz).unwrap();
                let layer = crate::laplacian_smoothing::LaplacianSmoothing {
                    vtx2idx: vtx2idx.clone(),
                    idx2vtx: idx2vtx.clone(),
                    lambda: 0.8,
                    num_iter: 10,
                };
                let grad = grad.apply_op1_no_bwd(&layer)?;
                let grad = (grad * 0.001)?;
                vtx2xyz.set(&vtx2xyz.sub(&(grad))?)?;
            }
            {
                let vtx2xyz = vtx2xyz.flatten_all()?.to_vec1::<f32>()?;
                let tri2vtx = tri2vtx.flatten_all()?.to_vec1::<u32>()?;
                del_msh_cpu::io_obj::save_tri2vtx_vtx2xyz(
                    format!("../target/hoge_{}.obj", itr),
                    &tri2vtx,
                    &vtx2xyz,
                    3,
                )?;
            }
        }
        Ok(())
    }
}
