#[test]
fn hoge() -> anyhow::Result<(), anyhow::Error> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("parent directory not found")
        .join("asset")
        .join("spot")
        .join("tet_mesh.npz");
    let file = std::fs::File::open(&path)?;
    let mut npz = ndarray_npy::NpzReader::new(file)?;
    let vtx2xyz: ndarray::Array2<f64> = npz.by_name("points.npy")?;
    let tet2vtx: ndarray::Array2<i32> = npz.by_name("tets.npy")?;
    println!("points shape = {:?}", vtx2xyz.dim()); // (N, 3)
    println!("tets shape   = {:?}", tet2vtx.dim()); // (M, 4)

    let (vtx2xyz, vtx2xyz_offset) = vtx2xyz.to_owned().into_raw_vec_and_offset();
    let (tet2vtx, tet2vtx_offset) = tet2vtx.to_owned().into_raw_vec_and_offset();
    assert_eq!(vtx2xyz_offset, Some(0));
    assert_eq!(tet2vtx_offset, Some(0));

    let tet2vtx = tet2vtx.iter().map(|v| *v as usize).collect::<Vec<_>>();
    println!("{}", tet2vtx.iter().min().unwrap());

    {
        let mut file = std::fs::File::create("../target/tetmesh.vtk").expect("file not found.");
        crate::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz, 3).unwrap();
        crate::io_vtk::write_vtk_cells(&mut file, crate::io_vtk::VtkElementType::TETRA, &tet2vtx)
            .unwrap();
    }

    let tet2tet = crate::uniform_mesh::elem2elem(
        &tet2vtx,
        4,
        &crate::elem2elem::TET_FACE2IDX,
        &crate::elem2elem::TET_IDX2NODE,
        vtx2xyz.len(),
    );

    let tri2vtx = crate::elem2elem::extract_boundary_mesh_for_uniform_mesh(
        &tet2vtx,
        4,
        &tet2tet,
        &crate::elem2elem::TET_FACE2IDX,
        &crate::elem2elem::TET_IDX2NODE,
    );

    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(
        "../target/tetmesh_boundary.obj",
        &tri2vtx,
        &vtx2xyz,
        3,
    )
    .unwrap();

    // crate::trimesh3::extend_avoid_intersection(&tri2vtx, &vtx2xyz, 0.01);

    Ok(())
}
