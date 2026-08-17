use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
//
use pyo3::Bound;

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    // topology
    m.add_function(wrap_pyfunction!(build_bvh_topology_topdown, m)?)?;
    m.add_function(wrap_pyfunction!(build_bvh_topology_morton, m)?)?;
    m.add_function(wrap_pyfunction!(shift_bvhnodes, m)?)?;
    // geometry
    m.add_function(wrap_pyfunction!(
        build_bvh_geometry_aabb_uniformmesh_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        build_bvh_geometry_aabb_uniformmesh_f64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(build_bvh_geometry_aabb_points_f32, m)?)?;
    m.add_function(wrap_pyfunction!(build_bvh_geometry_aabb_points_f64, m)?)?;
    Ok(())
}

// TODO make this function for uniforma mesh
#[pyo3::pyfunction]
fn build_bvh_topology_topdown<'a>(
    _py: pyo3::Python<'a>,
    tri2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>,
) -> Bound<'a, numpy::PyArray2<usize>> {
    assert!(tri2vtx.is_c_contiguous());
    assert!(vtx2xyz.is_c_contiguous());
    assert_eq!(vtx2xyz.shape()[1], 3);
    let tri2vtx = tri2vtx.as_slice().unwrap();
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    // change this to uniform mesh
    let bvhnodes = del_msh_cpu::bvhnodes_topdown_trimesh3::from_triangle_mesh(
        tri2vtx.as_chunks::<3>().0,
        vtx2xyz.as_chunks::<3>().0,
    );
    let bvhnodes = numpy::PyArray1::<usize>::from_slice(_py, bvhnodes.as_flattened());
    bvhnodes.reshape((bvhnodes.len() / 3, 3)).unwrap()
}

#[pyo3::pyfunction]
fn build_bvh_topology_morton<'a>(
    _py: pyo3::Python<'a>,
    vtx2xyz: numpy::PyReadonlyArray2<'a, f32>,
) -> Bound<'a, numpy::PyArray2<usize>> {
    let num_vtx = vtx2xyz.shape()[0];
    let num_dim = vtx2xyz.shape()[1];
    assert_eq!(vtx2xyz.shape(), [num_vtx, num_dim]);
    assert!(vtx2xyz.is_c_contiguous());
    let vtx2xyz = vtx2xyz.as_slice().unwrap();
    let mut idx2vtx = vec![0usize; num_vtx];
    let mut idx2morton = vec![0u32; num_vtx];
    let mut vtx2morton = vec![0u32; num_vtx];
    if num_dim == 3 {
        del_msh_cpu::mortons::sorted_morten_code3(
            &mut idx2vtx,
            &mut idx2morton,
            &mut vtx2morton,
            vtx2xyz.as_chunks::<3>().0,
            &del_geo_core::mat4_col_major::from_identity(),
        );
    } else if num_dim == 2 {
        del_msh_cpu::mortons::sorted_morten_code2(
            &mut idx2vtx,
            &mut idx2morton,
            &mut vtx2morton,
            vtx2xyz.as_chunks::<2>().0,
            &del_geo_core::mat3_col_major::from_identity(),
        );
    }
    let bvhnodes = numpy::PyArray2::<usize>::zeros(_py, (num_vtx * 2 - 1, 3), false);
    {
        let bvhnodes_slice = unsafe { bvhnodes.as_slice_mut().unwrap() };
        del_msh_cpu::bvhnodes_morton::update_bvhnodes(
            bvhnodes_slice.as_chunks_mut::<3>().0,
            &idx2vtx,
            &idx2morton,
        );
    }
    bvhnodes
}

#[pyo3::pyfunction]
#[allow(clippy::identity_op)]
fn shift_bvhnodes<'a>(
    _py: pyo3::Python<'a>,
    mut bvhnodes: numpy::PyReadwriteArray2<'a, usize>,
    node_offset: usize,
    idx_offset: usize,
) {
    assert!(bvhnodes.is_c_contiguous());
    let num_bvhnode = bvhnodes.shape()[0];
    assert_eq!(bvhnodes.shape()[1], 3);
    let bvhnodes = bvhnodes.as_slice_mut().unwrap();
    for i in 0..num_bvhnode {
        if bvhnodes[i * 3 + 0] != usize::MAX {
            bvhnodes[i * 3 + 0] += node_offset;
        }
        if bvhnodes[i * 3 + 2] == usize::MAX {
            bvhnodes[i * 3 + 1] += idx_offset;
            continue;
        } else {
            bvhnodes[i * 3 + 1] += node_offset;
            bvhnodes[i * 3 + 2] += node_offset;
        }
    }
}

fn build_bvh_geometry_aabb_uniformmesh<'a, T>(
    _py: pyo3::Python<'a>,
    mut bvhnode2aabb: numpy::PyReadwriteArray2<'a, T>,
    bvhnode2elem_tree: numpy::PyReadonlyArray2<'a, usize>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, T>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, T>,
) where
    T: numpy::Element + num_traits::Float,
{
    assert!(bvhnode2aabb.is_c_contiguous());
    assert!(bvhnode2elem_tree.is_c_contiguous());
    assert!(elem2vtx.is_c_contiguous());
    assert!(vtx2xyz0.is_c_contiguous());
    assert!(vtx2xyz1.is_c_contiguous());
    assert_eq!(bvhnode2elem_tree.shape()[0], bvhnode2aabb.shape()[0]);
    assert_eq!(bvhnode2elem_tree.shape()[1], 3);
    assert_eq!(bvhnode2aabb.shape()[1], 6);
    let aabbs = bvhnode2aabb.as_slice_mut().unwrap();
    let bvhnodes = bvhnode2elem_tree.as_slice().unwrap();
    let num_noel = elem2vtx.shape()[1];
    let elem2vtx = elem2vtx.as_slice().unwrap();
    let vtx2xyz1 = if vtx2xyz0.shape() == vtx2xyz1.shape() {
        Some(vtx2xyz1.as_slice().unwrap().as_chunks::<3>().0)
    } else {
        None
    };
    let vtx2xyz0 = vtx2xyz0.as_slice().unwrap().as_chunks::<3>().0;
    match num_noel {
        2 => del_msh_cpu::bvhnode2aabb3::update_for_uniform_mesh_with_bvh::<_, _, 2>(
            aabbs.as_chunks_mut::<6>().0,
            i_bvhnode_root,
            bvhnodes.as_chunks::<3>().0,
            elem2vtx.as_chunks::<2>().0,
            vtx2xyz0,
            vtx2xyz1,
        ),
        3 => del_msh_cpu::bvhnode2aabb3::update_for_uniform_mesh_with_bvh::<_, _, 3>(
            aabbs.as_chunks_mut::<6>().0,
            i_bvhnode_root,
            bvhnodes.as_chunks::<3>().0,
            elem2vtx.as_chunks::<3>().0,
            vtx2xyz0,
            vtx2xyz1,
        ),
        _ => todo!("unsupported num_noel: {}", num_noel),
    }
}

// 2D and 3D
#[pyo3::pyfunction]
fn build_bvh_geometry_aabb_uniformmesh_f32<'a>(
    _py: pyo3::Python<'a>,
    bvhnode2aabb: numpy::PyReadwriteArray2<'a, f32>,
    bvhnode2elem_tree: numpy::PyReadonlyArray2<'a, usize>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, f32>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, f32>,
) {
    build_bvh_geometry_aabb_uniformmesh::<f32>(
        _py,
        bvhnode2aabb,
        bvhnode2elem_tree,
        elem2vtx,
        vtx2xyz0,
        i_bvhnode_root,
        vtx2xyz1,
    );
}

#[pyo3::pyfunction]
fn build_bvh_geometry_aabb_uniformmesh_f64<'a>(
    _py: pyo3::Python<'a>,
    bvhnode2aabb: numpy::PyReadwriteArray2<'a, f64>,
    bvhnode2elem_tree: numpy::PyReadonlyArray2<'a, usize>,
    elem2vtx: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, f64>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, f64>,
) {
    build_bvh_geometry_aabb_uniformmesh::<f64>(
        _py,
        bvhnode2aabb,
        bvhnode2elem_tree,
        elem2vtx,
        vtx2xyz0,
        i_bvhnode_root,
        vtx2xyz1,
    );
}

fn build_bvh_geometry_aabb_points<'a, T>(
    _py: pyo3::Python<'a>,
    mut bvhnode2aabb: numpy::PyReadwriteArray2<'a, T>,
    bvhnode2vtx_tree: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, T>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, T>,
) where
    T: numpy::Element + num_traits::Float,
{
    assert!(bvhnode2aabb.is_c_contiguous());
    assert!(bvhnode2vtx_tree.is_c_contiguous());
    assert!(vtx2xyz0.is_c_contiguous());
    assert!(vtx2xyz1.is_c_contiguous());
    assert_eq!(bvhnode2vtx_tree.shape()[0], bvhnode2aabb.shape()[0]);
    assert_eq!(bvhnode2vtx_tree.shape()[1], 3);
    assert_eq!(bvhnode2aabb.shape()[1], 6);
    let aabbs = bvhnode2aabb.as_slice_mut().unwrap();
    let bvhnodes = bvhnode2vtx_tree.as_slice().unwrap();
    let vtx2xyz1 = if vtx2xyz0.shape() == vtx2xyz1.shape() {
        Some(vtx2xyz1.as_slice().unwrap().as_chunks::<3>().0)
    } else {
        None
    };
    let vtx2xyz0 = vtx2xyz0.as_slice().unwrap().as_chunks::<3>().0;
    del_msh_cpu::bvhnode2aabb3::update_for_points_with_bvh(
        aabbs.as_chunks_mut::<6>().0,
        i_bvhnode_root,
        bvhnodes.as_chunks::<3>().0,
        vtx2xyz0,
        vtx2xyz1,
    );
}

// 2D and 3D
#[pyo3::pyfunction]
fn build_bvh_geometry_aabb_points_f32<'a>(
    _py: pyo3::Python<'a>,
    bvhnode2aabb: numpy::PyReadwriteArray2<'a, f32>,
    bvhnode2vtx_tree: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, f32>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, f32>,
) {
    build_bvh_geometry_aabb_points::<f32>(
        _py,
        bvhnode2aabb,
        bvhnode2vtx_tree,
        vtx2xyz0,
        i_bvhnode_root,
        vtx2xyz1,
    );
}

#[pyo3::pyfunction]
fn build_bvh_geometry_aabb_points_f64<'a>(
    _py: pyo3::Python<'a>,
    bvhnode2aabb: numpy::PyReadwriteArray2<'a, f64>,
    bvhnode2vtx_tree: numpy::PyReadonlyArray2<'a, usize>,
    vtx2xyz0: numpy::PyReadonlyArray2<'a, f64>,
    i_bvhnode_root: usize,
    vtx2xyz1: numpy::PyReadonlyArray2<'a, f64>,
) {
    build_bvh_geometry_aabb_points::<f64>(
        _py,
        bvhnode2aabb,
        bvhnode2vtx_tree,
        vtx2xyz0,
        i_bvhnode_root,
        vtx2xyz1,
    );
}
