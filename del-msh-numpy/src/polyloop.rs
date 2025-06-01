use numpy::PyUntypedArrayMethods;
//
use pyo3::Bound;

pub fn add_functions(_py: pyo3::Python, m: &Bound<pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::PyModuleMethods;
    use pyo3::wrap_pyfunction;
    m.add_function(wrap_pyfunction!(polyloop2_area_f32, m)?)?;
    m.add_function(wrap_pyfunction!(polyloop2_area_f64, m)?)?;
    Ok(())
}

fn polyloop2_area<T>(vtx2xy: numpy::PyReadonlyArray2<T>) -> T
where
    T: numpy::Element + num_traits::Float + 'static + Copy + std::ops::AddAssign,
    f64: num_traits::AsPrimitive<T>,
{
    assert_eq!(vtx2xy.shape()[1], 2);
    let vtx2xy = vtx2xy.as_slice().unwrap();
    del_msh_cpu::polyloop2::area(vtx2xy)
}

#[pyo3::pyfunction]
fn polyloop2_area_f32(vtx2xy: numpy::PyReadonlyArray2<f32>) -> f32 {
    polyloop2_area::<f32>(vtx2xy)
}

#[pyo3::pyfunction]
fn polyloop2_area_f64(vtx2xy: numpy::PyReadonlyArray2<f64>) -> f64 {
    polyloop2_area::<f64>(vtx2xy)
}

// ----------------------------
