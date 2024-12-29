use candle_core::{CpuStorage, Layout, Tensor};

struct SortedMortonCode {}

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
        let vtx2pos = match vtx2pos {
            CpuStorage::F32(vtx2xyz) => vtx2xyz,
            _ => panic!(),
        };
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
}

struct BvhNodesFromSortedMortonCode {}

impl candle_core::InplaceOp2 for BvhNodesFromSortedMortonCode {
    fn name(&self) -> &'static str {
        "BvhNodes_FromSortedMorton"
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
        let morton_data = match morton_data {
            CpuStorage::U32(morton_data) => morton_data,
            _ => panic!(),
        };
        let (idx2vtx, idx2morton) = morton_data.split_at(num_vtx);
        let (idx2morton, _vtx2morton) = idx2morton.split_at(num_vtx);
        del_msh_core::bvhnodes_morton::update_bvhnodes(bvhnodes, idx2vtx, idx2morton);
        Ok(())
    }
}

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
            let mut rng = rand::thread_rng();
            (0..num_vtx * 3).map(|_| rng.gen::<f32>()).collect()
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
    sorted_morton_code.inplace_op2(&tri2center, &SortedMortonCode {})?;
    bvhnodes.inplace_op2(&sorted_morton_code, &BvhNodesFromSortedMortonCode {})?;
    Ok(())
}
