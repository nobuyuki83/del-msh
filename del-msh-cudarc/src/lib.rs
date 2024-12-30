// use cudarc::driver::{CudaDevice, CudaSlice};

#[cfg(feature = "cuda")]
pub mod elem2center;

#[cfg(feature = "cuda")]
pub mod vtx2xyz;

#[cfg(feature = "cuda")]
pub mod bvhnodes_morton;

#[cfg(feature = "cuda")]
pub mod bvhnode2aabb;

#[cfg(feature = "cuda")]
pub fn assert_equal_cpu_gpu(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    tri2vtx: &[u32],
    vtx2xyz: &[f32],
) -> anyhow::Result<()> {
    use cudarc::driver::DeviceSlice;
    let num_tri = tri2vtx.len() / 3;
    let tri2cntr = del_msh_core::elem2center::from_uniform_mesh_as_points(tri2vtx, 3, vtx2xyz, 3);
    let bvhnodes = del_msh_core::bvhnodes_morton::from_triangle_mesh(tri2vtx, vtx2xyz, 3);
    let aabb = del_msh_core::vtx2xyz::aabb3(&tri2cntr, 0f32);
    let transform_cntr2uni =
        del_geo_core::mat4_col_major::from_aabb3_fit_into_unit_preserve_asp(&aabb);
    // del_geo_core::mat4_col_major::from_aabb3_fit_into_unit(&aabb);
    let mut idx2tri = vec![0usize; num_tri];
    let mut idx2morton = vec![0u32; num_tri];
    let mut tri2morton = vec![0u32; num_tri];
    del_msh_core::bvhnodes_morton::sorted_morten_code3(
        &mut idx2tri,
        &mut idx2morton,
        &mut tri2morton,
        &tri2cntr,
        &transform_cntr2uni,
    );
    //  ----------------------------
    let tri2vtx_dev = dev.htod_copy(tri2vtx.to_vec())?;
    let vtx2xyz_dev = dev.htod_copy(vtx2xyz.to_vec())?;
    let mut tri2cntr_dev = dev.alloc_zeros::<f32>(num_tri * 3)?;
    crate::elem2center::tri2cntr_from_trimesh3(dev, &tri2vtx_dev, &vtx2xyz_dev, &mut tri2cntr_dev)?;
    {
        // check tri2cntr
        let tri2cntr_hst = dev.dtoh_sync_copy(&tri2cntr_dev)?;
        for i in 0..tri2cntr.len() {
            let diff = (tri2cntr[i] - tri2cntr_hst[i]).abs();
            // assert!(diff<=f32::EPSILON, "{} {}", diff, f32::EPSILON);
            assert_eq!(tri2cntr[i], tri2cntr_hst[i], "{}", diff);
        }
    }
    let aabb_dev = crate::vtx2xyz::to_aabb3(dev, &tri2cntr_dev)?;
    {
        let aabb_hst = dev.dtoh_sync_copy(&aabb_dev)?;
        assert_eq!(aabb_hst.len(), 6);
        for i in 0..6 {
            assert_eq!(aabb[i], aabb_hst[i], "{:?} {:?}", aabb, aabb_hst);
        }
    }
    // get aabb
    let mut tri2morton_dev = dev.alloc_zeros(num_tri)?;
    let transform_cntr2uni_dev = dev.htod_copy(transform_cntr2uni.to_vec())?;
    crate::bvhnodes_morton::vtx2morton(
        dev,
        &tri2cntr_dev,
        &transform_cntr2uni_dev,
        &mut tri2morton_dev,
    )?;
    {
        let tri2morton_hst = dev.dtoh_sync_copy(&tri2morton_dev)?;
        assert_eq!(tri2morton_hst.len(), num_tri);
        for i in 0..tri2morton.len() {
            assert_eq!(
                tri2morton_hst[i], tri2morton[i],
                "{} {}",
                tri2morton_hst[i], tri2morton[i]
            );
        }
    }
    let mut idx2tri_dev = dev.alloc_zeros(num_tri)?;
    del_cudarc::util::set_consecutive_sequence(dev, &mut idx2tri_dev)?;
    del_cudarc::sort_by_key_u32::radix_sort_by_key_u32(dev, &mut tri2morton_dev, &mut idx2tri_dev)?;
    let idx2morton_dev = tri2morton_dev;
    {
        let idx2tri_hst = dev.dtoh_sync_copy(&idx2tri_dev)?;
        assert_eq!(idx2tri.len(), idx2tri_hst.len());
        for i in 0..idx2tri_hst.len() {
            assert_eq!(idx2tri_hst[i], idx2tri[i] as u32);
        }
    }
    //let mut idx2morton_dev = dev.alloc_zeros(num_tri)?;
    //del_cudarc_util::util::permute(&dev, &mut idx2morton_dev, &idx2tri_dev, &tri2morton_dev)?;
    {
        let idx2morton_hst = dev.dtoh_sync_copy(&idx2morton_dev)?;
        for i in 0..idx2morton_hst.len() {
            // assert_eq!(idx2morton[i], idx2morton_hst[i] as u32);
            assert_eq!(
                idx2morton[i], idx2morton_hst[i],
                "{} {}",
                idx2morton[i], idx2morton_hst[i]
            );
        }
    }
    let mut bvhnodes_dev = dev.alloc_zeros((num_tri * 2 - 1) * 3)?;
    crate::bvhnodes_morton::from_sorted_morton_codes(
        dev,
        &mut bvhnodes_dev,
        &idx2morton_dev,
        &idx2tri_dev,
    )?;
    {
        let bvhnodes_hst = dev.dtoh_sync_copy(&bvhnodes_dev)?;
        for i in 0..bvhnodes_hst.len() {
            // assert_eq!(bvhnodes_hst[i], bvhnodes[i], "{} {} {}", i, bvhnodes[i], bvhnodes_hst[i]);
            if bvhnodes_hst[i] != bvhnodes[i] {
                println!("{} {} {}", i, bvhnodes[i], bvhnodes_hst[i]);
            }
        }
    }
    let bvhnode2aabb = del_msh_core::bvhnode2aabb3::from_uniform_mesh_with_bvh(
        0,
        &bvhnodes,
        Some((tri2vtx, 3)),
        vtx2xyz,
        None,
    );
    let mut bvhnode2aabb_dev = dev.alloc_zeros::<f32>(bvhnodes_dev.len() / 3 * 6)?;
    crate::bvhnode2aabb::from_trimesh3_with_bvhnodes(
        dev,
        &tri2vtx_dev,
        &vtx2xyz_dev,
        &bvhnodes_dev,
        &mut bvhnode2aabb_dev,
    )?;
    {
        let bvhnode2aabb_from_gpu = dev.dtoh_sync_copy(&bvhnode2aabb_dev)?;
        for i in 0..(num_tri * 2 - 1) * 6 {
            assert_eq!(bvhnode2aabb_from_gpu[i], bvhnode2aabb[i]);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn make_bvh_from_trimesh3(
    dev: &std::sync::Arc<cudarc::driver::CudaDevice>,
    bvhnodes: &cudarc::driver::CudaSlice<u32>,
    bvhnode2aabbiii: &cudarc::driver::CudaSlice<f32>,
    tri2vtx_dev: &cudarc::driver::CudaSlice<u32>,
    vtx2xyz_dev: &cudarc::driver::CudaSlice<f32>,
) -> anyhow::Result<()> {
    use cudarc::driver::DeviceSlice;
    let num_tri = tri2vtx_dev.len() / 3;
    let mut tri2cntr_dev = dev.alloc_zeros::<f32>(num_tri * 3)?;
    crate::elem2center::tri2cntr_from_trimesh3(dev, tri2vtx_dev, vtx2xyz_dev, &mut tri2cntr_dev)?;
    let aabb_dev = crate::vtx2xyz::to_aabb3(dev, &tri2cntr_dev)?;
    let aabb = dev.dtoh_sync_copy(&aabb_dev)?;
    let aabb = arrayref::array_ref!(&aabb, 0, 6);
    // get aabb
    let mut tri2morton_dev = dev.alloc_zeros(num_tri)?;
    let transform_cntr2uni =
        del_geo_core::mat4_col_major::from_aabb3_fit_into_unit_preserve_asp(aabb);
    let transform_cntr2uni_dev = dev.htod_copy(transform_cntr2uni.to_vec())?;
    crate::bvhnodes_morton::vtx2morton(
        dev,
        &tri2cntr_dev,
        &transform_cntr2uni_dev,
        &mut tri2morton_dev,
    )?;
    let mut idx2tri_dev = dev.alloc_zeros(num_tri)?;
    del_cudarc::util::set_consecutive_sequence(dev, &mut idx2tri_dev)?;
    del_cudarc::sort_by_key_u32::radix_sort_by_key_u32(dev, &mut tri2morton_dev, &mut idx2tri_dev)?;
    let idx2morton_dev = tri2morton_dev;
    //let mut idx2morton_dev = dev.alloc_zeros(num_tri)?;
    //del_cudarc_util::util::permute(&dev, &mut idx2morton_dev, &idx2tri_dev, &tri2morton_dev)?;
    // let mut bvhnodes_dev = dev.alloc_zeros((num_tri * 2 - 1) * 3)?;
    crate::bvhnodes_morton::from_sorted_morton_codes(dev, bvhnodes, &idx2morton_dev, &idx2tri_dev)?;
    // let mut bvhnode2aabb_dev = dev.alloc_zeros::<f32>(bvhnodes_dev.len() / 3 * 6)?;
    crate::bvhnode2aabb::from_trimesh3_with_bvhnodes(
        dev,
        tri2vtx_dev,
        vtx2xyz_dev,
        &bvhnodes,
        bvhnode2aabbiii,
    )?;
    Ok(())
}
