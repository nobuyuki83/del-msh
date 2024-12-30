use candle_core::{CpuStorage, Layout, Shape, Tensor};

pub struct Layer {
    pub vtxl2xy: Vec<f32>,
    pub vtxv2info: Vec<[usize; 4]>,
}

impl candle_core::CustomOp1 for crate::voronoi2::Layer {
    fn name(&self) -> &'static str {
        "site2_to_volonoi2"
    }

    #[allow(clippy::identity_op)]
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (_num_site, num_dim) = layout.shape().dims2()?;
        assert_eq!(num_dim, 2);
        let site2xy = storage.as_slice::<f32>()?;
        let num_vtxv = self.vtxv2info.len();
        let mut vtxv2xy = vec![0f32; num_vtxv * 2];
        for i_vtxv in 0..num_vtxv {
            let cc = del_msh_nalgebra::voronoi2::position_of_voronoi_vertex(
                &self.vtxv2info[i_vtxv],
                &self.vtxl2xy,
                site2xy,
            );
            vtxv2xy[i_vtxv * 2 + 0] = cc.x;
            vtxv2xy[i_vtxv * 2 + 1] = cc.y;
        }
        let shape = candle_core::Shape::from((num_vtxv, 2));
        let storage = candle_core::WithDType::to_cpu_storage_owned(vtxv2xy);
        Ok((storage, shape))
    }

    /// This function takes as argument the argument `arg` used in the forward pass, the result
    /// produced by the forward operation `res` and the gradient of the result `grad_res`.
    /// The function should return the gradient of the argument.
    #[allow(clippy::identity_op)]
    fn bwd(
        &self,
        site2xy: &Tensor,
        _vtxv2xy: &Tensor,
        dw_vtxv2xy: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let (num_site, two) = site2xy.shape().dims2()?;
        assert_eq!(two, 2);
        let (num_vtxv, two) = _vtxv2xy.shape().dims2()?;
        assert_eq!(two, 2);
        use std::ops::Deref;
        let site2xy = site2xy.storage_and_layout().0;
        let site2xy = match site2xy.deref() {
            candle_core::Storage::Cpu(site2xy) => site2xy.as_slice::<f32>()?,
            _ => {
                panic!()
            }
        };
        //
        let dw_vtxv2xy = dw_vtxv2xy.storage_and_layout().0;
        let dw_vtxv2xy = match dw_vtxv2xy.deref() {
            candle_core::Storage::Cpu(dw_vtxv2xy) => dw_vtxv2xy.as_slice::<f32>()?,
            _ => {
                panic!()
            }
        };
        //
        let mut dw_site2xy = vec![0f32; num_site * 2];
        for i_vtxv in 0..num_vtxv {
            let info = self.vtxv2info[i_vtxv];
            if info[1] == usize::MAX {
                // this vtxv is one of vtxl
                assert!(info[0] < self.vtxl2xy.len() / 2);
            } else if info[3] == usize::MAX {
                // intersection of loop edge and two voronoi
                let num_vtxl = self.vtxl2xy.len() / 2;
                assert!(info[0] < num_vtxl);
                let i1_loop = info[0];
                let i2_loop = (i1_loop + 1) % num_vtxl;
                let l1 = del_msh_core::vtx2xy::to_navec2(&self.vtxl2xy, i1_loop);
                let l2 = del_msh_core::vtx2xy::to_navec2(&self.vtxl2xy, i2_loop);
                let i0_site = info[1];
                let i1_site = info[2];
                let s0 = del_msh_core::vtx2xy::to_navec2(site2xy, i0_site);
                let s1 = del_msh_core::vtx2xy::to_navec2(site2xy, i1_site);
                let (_r, drds0, drds1) = del_geo_nalgebra::line2::dw_intersection_against_bisector(
                    &l1,
                    &(l2 - l1),
                    &s0,
                    &s1,
                );
                let dv = del_msh_core::vtx2xy::to_navec2(dw_vtxv2xy, i_vtxv);
                {
                    let ds0 = drds0.transpose() * dv;
                    dw_site2xy[i0_site * 2 + 0] += ds0.x;
                    dw_site2xy[i0_site * 2 + 1] += ds0.y;
                }
                {
                    let ds1 = drds1.transpose() * dv;
                    dw_site2xy[i1_site * 2 + 0] += ds1.x;
                    dw_site2xy[i1_site * 2 + 1] += ds1.y;
                }
            } else {
                // circumference of three voronoi vtx
                let idx_site = [info[1], info[2], info[3]];
                let s0 = del_msh_core::vtx2xy::to_navec2(site2xy, idx_site[0]);
                let s1 = del_msh_core::vtx2xy::to_navec2(site2xy, idx_site[1]);
                let s2 = del_msh_core::vtx2xy::to_navec2(site2xy, idx_site[2]);
                let (_v, dvds) = del_geo_nalgebra::tri2::wdw_circumcenter(&s0, &s1, &s2);
                let dv = del_msh_core::vtx2xy::to_navec2(dw_vtxv2xy, i_vtxv);
                for i_node in 0..3 {
                    let ds0 = dvds[i_node].transpose() * dv;
                    let is0 = idx_site[i_node];
                    dw_site2xy[is0 * 2 + 0] += ds0.x;
                    dw_site2xy[is0 * 2 + 1] += ds0.y;
                }
            }
        }
        let dw_site2xy = Tensor::from_vec(
            dw_site2xy,
            candle_core::Shape::from((num_site, 2)),
            &candle_core::Device::Cpu,
        )?;
        Ok(Some(dw_site2xy))
    }
}

#[test]
fn test_backward() -> anyhow::Result<()> {
    use rand::SeedableRng;
    // use rand::Rng;
    let mut reng = rand_chacha::ChaChaRng::seed_from_u64(0);
    let vtxl2xy = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    // let mut reng = rand::thread_rng();
    let site2xy0 =
        del_msh_core::sampling::poisson_disk_sampling_from_polyloop2(&vtxl2xy, 0.15, 10, &mut reng);
    let site2xy0 = {
        candle_core::Var::from_slice(
            &site2xy0,
            candle_core::Shape::from((site2xy0.len() / 2, 2)),
            &candle_core::Device::Cpu,
        )
        .unwrap()
    };
    let (vtxv2xy0, voronoi_info0) = crate::voronoi2::voronoi(&vtxl2xy, &site2xy0, |_| true);
    {
        // output to obj file
        let mut vtx2xy = vtxv2xy0.clone().flatten_all()?.to_vec1::<f32>()?;
        let site2xy = site2xy0.clone().flatten_all()?.to_vec1::<f32>()?;
        vtx2xy.extend(site2xy);
        let edge2vtxv = del_msh_core::edge2vtx::from_polygon_mesh(
            &voronoi_info0.site2idx,
            &voronoi_info0.idx2vtxv,
            vtx2xy.len() / 2,
        );
        let _ = del_msh_core::io_obj::save_edge2vtx_vtx2xyz(
            "target/voronoi0.obj",
            &edge2vtxv,
            &vtx2xy,
            2,
        );
    }
    let vtxv2xygoal =
        candle_core::Tensor::randn(1f32, 1f32, vtxv2xy0.shape(), &candle_core::Device::Cpu)?;
    let loss0 = vtxv2xy0.mul(&vtxv2xygoal)?.sum_all()?;
    let grad = loss0.backward()?;
    let loss0 = loss0.to_vec0::<f32>()?;
    let dw_site2xy = grad.get(&site2xy0).unwrap();
    let dw_site2xy = dw_site2xy.flatten_all()?.to_vec1::<f32>()?;
    let eps = 1.0e-4;
    let num_site = site2xy0.dims2()?.0;
    for i_site in 0..num_site {
        for i_dim in 0..2 {
            let mut site2xy1 = site2xy0.clone().flatten_all()?.to_vec1::<f32>()?;
            site2xy1[i_site * 2 + i_dim] += eps;
            let site2xy1 = candle_core::Tensor::from_vec(
                site2xy1,
                site2xy0.shape(),
                &candle_core::Device::Cpu,
            )?;
            let site2_to_voronoi2 = Layer {
                vtxl2xy: vtxl2xy.clone(),
                vtxv2info: voronoi_info0.vtxv2info.clone(),
            };
            let vtxv2xy1 = site2xy1.apply_op1(site2_to_voronoi2)?;
            let loss1 = vtxv2xy1.mul(&vtxv2xygoal)?.sum_all()?;
            let loss1 = loss1.to_vec0::<f32>()?;
            let val0 = (loss1 - loss0) / eps;
            let val1 = dw_site2xy[i_site * 2 + i_dim];
            assert!((val0 - val1).abs() < 5.0e-2, "diff {} {}", val0, val1);
        }
    }
    Ok(())
}

#[derive(Default)]
pub struct VoronoiInfo {
    pub site2idx: Vec<usize>,
    pub idx2vtxv: Vec<usize>,
    pub idx2site: Vec<usize>,
    pub vtxv2info: Vec<[usize; 4]>,
}

pub fn voronoi<F>(
    vtxl2xy: &[f32],
    site2xy: &candle_core::Tensor,
    site2isalive: F,
) -> (Tensor, VoronoiInfo)
where
    F: Fn(usize) -> bool,
{
    let site2cell = del_msh_nalgebra::voronoi2::voronoi_cells(
        vtxl2xy,
        &site2xy.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &site2isalive,
    );
    let voronoi_mesh = del_msh_nalgebra::voronoi2::indexing(&site2cell);
    let site2_to_voronoi2 = crate::voronoi2::Layer {
        vtxl2xy: Vec::<f32>::from(vtxl2xy),
        vtxv2info: voronoi_mesh.vtxv2info.clone(),
    };
    let vtxv2xy = site2xy.apply_op1(site2_to_voronoi2).unwrap();
    let idx2site = del_msh_core::elem2elem::from_polygon_mesh(
        &voronoi_mesh.site2idx,
        &voronoi_mesh.idx2vtxv,
        vtxv2xy.dims2().unwrap().0,
    );
    let vi = VoronoiInfo {
        site2idx: voronoi_mesh.site2idx,
        idx2vtxv: voronoi_mesh.idx2vtxv,
        vtxv2info: voronoi_mesh.vtxv2info,
        idx2site,
    };
    (vtxv2xy, vi)
}

pub fn loss_lloyd(
    elem2idx: &[usize],
    idx2vtx: &[usize],
    site2xy: &candle_core::Tensor,
    vtxv2xy: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let polygonmesh2_to_cogs = crate::polygonmesh2_to_cogs::Layer {
        elem2idx: Vec::from(elem2idx),
        idx2vtx: Vec::from(idx2vtx),
    };
    let site2cogs = vtxv2xy.apply_op1(polygonmesh2_to_cogs)?;
    site2xy.sub(&site2cogs)?.sqr().unwrap().sum_all()
}
