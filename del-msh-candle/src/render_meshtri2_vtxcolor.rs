#[allow(unused_imports)]
use candle_core::{CpuStorage, Device, Layout, Shape, Tensor};
use std::ops::Deref;

pub struct Layer {
    tri2vtx: Tensor,
    vtx2xy: Tensor,
    pix2tri: Tensor,
    img_shape: (usize, usize),  // (width, height)
    transform_xy2pix: [f32; 9], // transform column major
}

impl candle_core::CustomOp1 for Layer {
    fn name(&self) -> &'static str {
        "render"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (num_vtx, two) = self.vtx2xy.shape().dims2()?;
        assert_eq!(two, 2);
        let (num_vtx1, num_channel) = layout.shape().dims2()?;
        // dbg!(num_dim);
        assert_eq!(num_vtx, num_vtx1);
        let vtx2color = storage.as_slice::<f32>()?;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let vtx2xy = self.vtx2xy.storage_and_layout().0;
        let vtx2xy = match vtx2xy.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let img2tri = self.pix2tri.storage_and_layout().0;
        let img2tri = match img2tri.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        let mut img = vec![0f32; self.img_shape.0 * self.img_shape.1];
        let transform_pix2xy =
            del_geo_core::mat3_col_major::try_inverse(&self.transform_xy2pix).unwrap();
        for i_h in 0..self.img_shape.1 {
            for i_w in 0..self.img_shape.0 {
                let i_tri = img2tri[i_h * self.img_shape.0 + i_w];
                if i_tri == u32::MAX {
                    continue;
                }
                let p_xy = del_geo_core::mat3_col_major::transform_homogeneous::<f32>(
                    &transform_pix2xy,
                    &[i_w as f32 + 0.5, i_h as f32 + 0.5],
                )
                .unwrap();
                let (p0, p1, p2) =
                    del_msh_cpu::trimesh2::to_corner_points(tri2vtx, vtx2xy, i_tri as usize);
                let Some((r0, r1, r2)) =
                    del_geo_core::tri2::barycentric_coords(&p0, &p1, &p2, &p_xy)
                else {
                    continue;
                };
                let i_tri = i_tri as usize;
                let iv0: usize = tri2vtx[i_tri * 3] as usize;
                let iv1: usize = tri2vtx[i_tri * 3 + 1] as usize;
                let iv2: usize = tri2vtx[i_tri * 3 + 2] as usize;
                for i_channel in 0..num_channel {
                    let c0 = vtx2color[iv0 * num_channel + i_channel];
                    let c1 = vtx2color[iv1 * num_channel + i_channel];
                    let c2 = vtx2color[iv2 * num_channel + i_channel];
                    img[(i_h * self.img_shape.0 + i_w) * num_channel + i_channel] =
                        r0 * c0 + r1 * c1 + r2 * c2;
                }
            }
        }
        let shape = candle_core::Shape::from((self.img_shape.0, self.img_shape.1, num_channel));
        let storage = candle_core::WithDType::to_cpu_storage_owned(img);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(
        &self,
        vtx2color: &Tensor,
        pix2color: &Tensor,
        dw_pix2color: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (num_vtx, num_channels) = vtx2color.shape().dims2()?;
        let (height, width, _num_channels) = pix2color.shape().dims3()?;
        assert_eq!(num_channels, _num_channels);
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_tri2vtx) => cpu_tri2vtx.as_slice::<u32>()?,
            _ => panic!(),
        };
        let vtx2xy = self.vtx2xy.storage_and_layout().0;
        let vtx2xy = match vtx2xy.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        let pix2tri = self.pix2tri.storage_and_layout().0;
        let pix2tri = match pix2tri.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        assert_eq!(vtx2xy.len(), num_vtx * 2);
        let dw_pix2color = dw_pix2color.storage_and_layout().0;
        let dw_pix2color = match dw_pix2color.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<f32>()?,
            _ => panic!(),
        };
        assert_eq!(dw_pix2color.len(), height * width * num_channels);
        //
        let mut dw_vtx2color = vec![0f32; num_vtx * num_channels];
        let transform_pix2xy =
            del_geo_core::mat3_col_major::try_inverse(&self.transform_xy2pix).unwrap();
        for i_h in 0..height {
            for i_w in 0..width {
                let i_tri = pix2tri[i_h * self.img_shape.0 + i_w];
                if i_tri == u32::MAX {
                    continue;
                }
                let p_xy = del_geo_core::mat3_col_major::transform_homogeneous(
                    &transform_pix2xy,
                    &[i_w as f32 + 0.5, i_h as f32 + 0.5],
                )
                .unwrap();
                let (p0, p1, p2) =
                    del_msh_cpu::trimesh2::to_corner_points(tri2vtx, vtx2xy, i_tri as usize);
                let Some((r0, r1, r2)) =
                    del_geo_core::tri2::barycentric_coords(&p0, &p1, &p2, &p_xy)
                else {
                    continue;
                };
                let i_tri = i_tri as usize;
                let iv0 = tri2vtx[i_tri * 3 + 0] as usize;
                let iv1 = tri2vtx[i_tri * 3 + 1] as usize;
                let iv2 = tri2vtx[i_tri * 3 + 2] as usize;
                for i_ch in 0..num_channels {
                    let dw_color = dw_pix2color[(i_h * width + i_w) * num_channels + i_ch];
                    dw_vtx2color[iv0 * num_channels + i_ch] += dw_color * r0;
                    dw_vtx2color[iv1 * num_channels + i_ch] += dw_color * r1;
                    dw_vtx2color[iv2 * num_channels + i_ch] += dw_color * r2;
                }
            }
        }
        let dw_vtx2color = candle_core::Tensor::from_vec(
            dw_vtx2color,
            candle_core::Shape::from((num_vtx, num_channels)),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_vtx2color))
    }
}

#[test]
fn test_optimize_vtxcolor() -> anyhow::Result<()> {
    let img_trg = {
        let (img_trg, (width, height), depth) =
            del_canvas::load_image_as_float_array("../asset/tesla.png").unwrap();
        let img_trg: Vec<f32> = img_trg.chunks(3).map(|v| v[0]).collect(); // grayscale
        Tensor::from_vec(
            img_trg,
            candle_core::Shape::from((height, width, 1)),
            &Device::Cpu,
        )
        .unwrap()
    };
    let img_shape = (img_trg.dims3().unwrap().1, img_trg.dims3().unwrap().0);
    // transformation from xy to pixel coordinate
    let transform_xy2pix: [f32; 9] =
        del_geo_core::mat3_col_major::transform_world2pix_ortho_preserve_asp(
            &img_shape,
            &[0.0, 0.0, 1.0, 1.0],
        );
    let (tri2vtx, vtx2xyz) = del_msh_cpu::trimesh2_dynamic::meshing_from_polyloop2::<u32, f32>(
        &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        0.03,
        0.03,
    );

    // ------------------
    // below: candle
    let num_vtx = vtx2xyz.len() / 2;
    let vtx2xy = Tensor::from_vec(
        vtx2xyz,
        candle_core::Shape::from((num_vtx, 2)),
        &Device::Cpu,
    )
    .unwrap();
    let num_tri = tri2vtx.len() / 3;
    let tri2vtx = Tensor::from_vec(
        tri2vtx,
        candle_core::Shape::from((num_tri, 3)),
        &Device::Cpu,
    )
    .unwrap();
    let pix2tri = {
        let (bvhnodes, aabbs) = {
            let bvhdata = crate::bvhnode2aabb::BvhForTriMesh::new(num_tri, 2, &Device::Cpu)?;
            bvhdata.compute(&tri2vtx, &vtx2xy)?;
            (bvhdata.bvhnodes, bvhdata.bvhnode2aabb)
        };
        let pix2tri = crate::raycast_trimesh::raycast2(
            &tri2vtx,
            &vtx2xy,
            &bvhnodes,
            &aabbs,
            &img_shape,
            &transform_xy2pix,
        )?;
        pix2tri
    };
    let vtx2color = {
        use rand::Rng;
        let mut rng = rand::rng();
        let vals: Vec<f32> = (0..num_vtx).map(|_| rng.random::<f32>()).collect();
        candle_core::Var::from_vec(
            vals,
            candle_core::Shape::from((num_vtx, 1)),
            &candle_core::Device::Cpu,
        )
        .unwrap()
    };
    dbg!(&vtx2color.shape());

    let now = std::time::Instant::now();
    for i_itr in 0..100 {
        let render = Layer {
            tri2vtx: tri2vtx.clone(),
            vtx2xy: vtx2xy.clone(),
            pix2tri: pix2tri.clone(),
            img_shape,
            transform_xy2pix,
        };
        let img_out = vtx2color.apply_op1(render)?;
        let diff = img_trg.sub(&img_out).unwrap().sqr()?.sum_all()?;
        let grad = diff.backward()?;
        let dw_vtx2color = grad.get(&vtx2color).unwrap();
        if i_itr % 10 == 0 {
            let img_out_vec: Vec<f32> = img_out.flatten_all()?.to_vec1()?;
            del_canvas::write_png_from_float_image_grayscale(
                format!(
                    "../target/render_meshtri2_vtxcolor-test_optimize_vtxcolor_{}.png",
                    i_itr
                ),
                img_shape,
                &img_out_vec,
            )?;
        }
        let _ = vtx2color.set(&vtx2color.as_tensor().sub(&(dw_vtx2color * 0.003)?)?);
    }
    println!("Elapsed: {:.2?}", now.elapsed());
    Ok(())
}
