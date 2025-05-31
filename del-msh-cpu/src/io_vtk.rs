//! method for VTK files

pub enum VtkElementType {
    TRIANGLE = 5,
    QUAD = 9,
    TETRA = 10,
    HEXAHEDRON = 12,
    WEDGE,
    PYRAMID,
}

pub fn write_vtk_points<T>(
    file: &mut std::fs::File,
    name: &str,
    vtx2xyz: &[T],
    ndim: usize,
) -> std::io::Result<()>
where
    T: std::fmt::Display,
{
    use std::io::Write;
    let np = vtx2xyz.len() / ndim;
    writeln!(file, "# vtk DataFile Version 2.0")?;
    writeln!(file, "{}", name)?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(file, "POINTS {} float", np)?;
    if ndim == 3 {
        for xyz in vtx2xyz.chunks(3) {
            writeln!(file, "{} {} {}", xyz[0], xyz[1], xyz[2])?;
        }
    } else if ndim == 2 {
        for xy in vtx2xyz.chunks(2) {
            writeln!(file, "{} {}", xy[0], xy[1])?;
        }
    } else {
        panic!();
    }
    Ok(())
}

pub fn write_vtk_cells(
    file: &mut std::fs::File,
    vtk_elem_type: VtkElementType,
    elem2vtx: &[usize],
) -> std::io::Result<()> {
    let num_node = match vtk_elem_type {
        VtkElementType::TRIANGLE => 3,
        VtkElementType::QUAD => 4,
        VtkElementType::TETRA => 4,
        VtkElementType::HEXAHEDRON => 8,
        _ => {
            panic!();
        }
    };
    let nelem = elem2vtx.len() / num_node;
    use std::io::Write;
    writeln!(file, "CELLS {} {}", nelem, nelem * (num_node + 1))?;
    for av in elem2vtx.chunks(num_node) {
        write!(file, "{}", num_node)?;
        for v in av {
            write!(file, " {}", v)?;
        }
        writeln!(file)?;
    }
    writeln!(file, "CELL_TYPES {}", nelem)?;
    {
        let id_elem = vtk_elem_type as usize;
        for _ in 0..nelem {
            writeln!(file, "{}", id_elem)?;
        }
    }
    Ok(())
}

pub fn write_vtk_data_point_scalar<T>(
    file: &mut std::fs::File,
    vtx2data: &[T],
    num_vtx: usize,
    num_stride: usize,
) -> std::io::Result<()>
where
    T: std::fmt::Display,
{
    use std::io::Write;
    writeln!(file, "SCALARS pointvalue float 1")?;
    writeln!(file, "LOOKUP_TABLE default")?;
    for ip in 0..num_vtx {
        writeln!(file, "{}", vtx2data[ip * num_stride])?;
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::io_vtk::VtkElementType;

    #[test]
    fn trimesh3_scalardata() {
        let (tri2vtx, vtx2xyz) = crate::trimesh3_primitive::hemisphere_zup::<f64>(1., 16, 32);
        let mut file = std::fs::File::create("../target/trimesh3.vtk").expect("file not found.");
        let _ = crate::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz, 3);
        let _ = crate::io_vtk::write_vtk_cells(&mut file, VtkElementType::TRIANGLE, &tri2vtx);
        let vtx2data = {
            let mut vtx2data = Vec::<f64>::with_capacity(vtx2xyz.len() / 3);
            for i_vtx in 0..vtx2xyz.len() / 3 {
                let z = vtx2xyz[i_vtx * 3 + 2];
                vtx2data.push(z.powi(3));
            }
            vtx2data
        };
        use std::io::Write;
        let _ = writeln!(file, "POINT_DATA {}", vtx2xyz.len() / 3);
        let _ =
            crate::io_vtk::write_vtk_data_point_scalar(&mut file, &vtx2data, vtx2xyz.len() / 3, 1);
    }
}
