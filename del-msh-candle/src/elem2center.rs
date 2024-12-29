use candle_core::{CpuStorage, Layout};

pub struct Layer {}

impl candle_core::InplaceOp3 for Layer {
    fn name(&self) -> &'static str {
        "elem2center"
    }

    fn cpu_fwd(
        &self,
        elem2center: &mut CpuStorage,
        _l1: &Layout,
        elem2vtx: &CpuStorage,
        _l2: &Layout,
        vtx2xyz: &CpuStorage,
        _l3: &Layout,
    ) -> candle_core::Result<()> {
        let num_node = _l2.dims()[1];
        let num_dim = _l3.dims()[1];
        let elem2vtx = match elem2vtx {
            CpuStorage::U32(vec) => vec,
            _ => panic!(),
        };
        let vtx2xyz = match vtx2xyz {
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
    let elem2center = candle_core::Tensor::zeros(
        (num_tri, 3),
        candle_core::DType::F32,
        &candle_core::Device::Cpu,
    )?;
    elem2center.inplace_op3(&tri2vtx, &vtx2xyz, &Layer {})?;
    Ok(())
}
