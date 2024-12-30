use candle_core::{CpuStorage, DType, Device, Layout, Tensor};

pub struct Layer {
    pub tri2vtx: Tensor,
}

impl candle_core::InplaceOp3 for Layer {
    fn name(&self) -> &'static str {
        "bvhnode2aabb"
    }

    fn cpu_fwd(
        &self,
        bvhnode2aabb: &mut CpuStorage,
        l_bvhnode2aabb: &Layout,
        bvhnodes: &CpuStorage,
        l_bvhnodes: &Layout,
        vtx2xyz: &CpuStorage,
        l_vtx2xyz: &Layout,
    ) -> candle_core::Result<()> {
        assert_eq!(l_bvhnode2aabb.dims().len(), 2);
        let num_bvhnode = l_bvhnode2aabb.dims()[0];
        assert_eq!(l_bvhnode2aabb.dim(1)?, 6);
        assert_eq!(l_bvhnodes.dim(0)?, num_bvhnode);
        assert_eq!(l_bvhnodes.dim(1)?, 3);
        use std::ops::Deref;
        let num_dim = l_vtx2xyz.dim(1)?;
        let bvhnodes = match bvhnodes {
            CpuStorage::U32(a) => a,
            _ => panic!(),
        };
        let vtx2xyz = match vtx2xyz {
            CpuStorage::F32(a) => a,
            _ => panic!(),
        };
        let bvhnode2aabb = match bvhnode2aabb {
            CpuStorage::F32(a) => a,
            _ => panic!(),
        };
        let num_node = self.tri2vtx.dims2()?.1;
        let tri2vtx = self.tri2vtx.storage_and_layout().0;
        let tri2vtx = match tri2vtx.deref() {
            candle_core::Storage::Cpu(cpu_storage) => cpu_storage.as_slice::<u32>()?,
            _ => panic!(),
        };
        match num_dim {
            3 => del_msh_core::bvhnode2aabb3::update_for_uniform_mesh_with_bvh(
                bvhnode2aabb,
                0,
                bvhnodes,
                Some((tri2vtx, num_node)),
                vtx2xyz,
                None,
            ),
            2 => del_msh_core::bvhnode2aabb2::update_for_uniform_mesh_with_bvh(
                bvhnode2aabb,
                0,
                bvhnodes,
                Some((tri2vtx, num_node)),
                vtx2xyz,
                None,
            ),
            _ => panic!(),
        }
        Ok(())
    }
}

pub struct BvhForTriMesh {
    tri2center: Tensor,
    sorted_morton_code: Tensor,
    bvhnodes: Tensor,
    bvhnode2aabb: Tensor,
}

impl BvhForTriMesh {
    pub fn new(num_tri: usize, num_dim: usize) -> anyhow::Result<Self> {
        let tri2center = candle_core::Tensor::zeros((num_tri, num_dim), DType::F32, &Device::Cpu)?;
        let sorted_morton_code = candle_core::Tensor::zeros(num_tri * 3, DType::U32, &Device::Cpu)?;
        let num_bvhnodes = num_tri * 2 - 1;
        let bvhnodes = candle_core::Tensor::zeros((num_bvhnodes, 3), DType::U32, &Device::Cpu)?;
        let bvhnode2aabb =
            candle_core::Tensor::zeros((num_bvhnodes, num_dim * 2), DType::F32, &Device::Cpu)?;
        let res = BvhForTriMesh {
            tri2center,
            sorted_morton_code,
            bvhnodes,
            bvhnode2aabb,
        };
        Ok(res)
    }

    pub fn compute(&self, tri2vtx: &Tensor, vtx2xyz: &Tensor) -> candle_core::Result<()> {
        self.tri2center
            .inplace_op3(tri2vtx, vtx2xyz, &crate::elem2center::Layer {})?;
        self.sorted_morton_code.inplace_op2(
            &self.tri2center,
            &crate::bvhnodes_morton::SortedMortonCode {},
        )?;
        self.bvhnodes.inplace_op2(
            &self.sorted_morton_code,
            &crate::bvhnodes_morton::BvhNodesFromSortedMortonCode {},
        )?;
        let layer = Layer {
            tri2vtx: tri2vtx.clone(),
        };
        self.bvhnode2aabb
            .inplace_op3(&self.bvhnodes, vtx2xyz, &layer)?;
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
    let bvhdata = BvhForTriMesh::new(num_tri, 3)?;
    bvhdata.compute(&tri2vtx, &vtx2xyz)?;
    Ok(())
}
