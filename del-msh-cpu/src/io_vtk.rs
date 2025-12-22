//! method for VTK files

/// "#[repr(u32)]" is for "as" operator: t = VtkElementType::TETRA as u32 // t == 5
#[repr(u32)]
pub enum VtkElementType {
    TRIANGLE = 5,
    QUAD = 9,
    TETRA = 10,
    HEXAHEDRON = 12,
    WEDGE = 13,
    PYRAMID = 14,
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
    writeln!(file, "{name}")?;
    writeln!(file, "ASCII")?;
    writeln!(file, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(file, "POINTS {np} float")?;
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
        VtkElementType::PYRAMID => 5,
        VtkElementType::WEDGE => 6,
    };
    let num_elem = elem2vtx.len() / num_node;
    assert_eq!(elem2vtx.len(), num_elem * num_node);
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "CELLS {} {}", num_elem, num_elem * (num_node + 1))?;
    for av in elem2vtx.chunks(num_node) {
        write!(writer, "{num_node}")?;
        for v in av {
            write!(writer, " {v}")?;
        }
        writeln!(writer)?;
    }
    writeln!(writer, "CELL_TYPES {num_elem}")?;
    {
        let id_elem = vtk_elem_type as usize;
        for _ in 0..num_elem {
            writeln!(writer, "{id_elem}")?;
        }
    }
    Ok(())
}

pub fn write_vtk_cells_mix<IDX>(
    file: &mut std::fs::File,
    tet2vtx: &[IDX],
    pyramid2vtx: &[IDX],
    prism2vtx: &[IDX],
) -> std::io::Result<()>
where
    IDX: num_traits::PrimInt + std::fmt::Display,
{
    let num_tet = tet2vtx.len() / 4;
    let num_pyramid = pyramid2vtx.len() / 5;
    let num_prism = prism2vtx.len() / 6;
    let num_elem = num_tet + num_pyramid + num_prism;
    let num_idx = 5 * num_tet + 6 * num_pyramid + 7 * num_prism;
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(file);
    writeln!(writer, "CELLS {num_elem} {num_idx}")?;
    for av in tet2vtx.chunks(4) {
        write!(writer, "4")?;
        av.iter().for_each(|v| write!(writer, " {v}").unwrap());
        writeln!(writer)?;
    }
    for av in pyramid2vtx.chunks(5) {
        write!(writer, "5")?;
        av.iter().for_each(|v| write!(writer, " {v}").unwrap());
        writeln!(writer)?;
    }
    for av in prism2vtx.chunks(6) {
        write!(writer, "6")?;
        av.iter().for_each(|v| write!(writer, " {v}").unwrap());
        writeln!(writer)?;
    }
    writeln!(writer, "CELL_TYPES {num_elem}")?;
    for _ in 0..num_tet {
        writeln!(writer, "{}", VtkElementType::TETRA as u32).unwrap();
    }
    for _ in 0..num_pyramid {
        writeln!(writer, "{}", VtkElementType::PYRAMID as u32).unwrap();
    }
    for _ in 0..num_prism {
        writeln!(writer, "{}", VtkElementType::WEDGE as u32).unwrap();
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
        crate::io_vtk::write_vtk_points(&mut file, "hoge", &vtx2xyz, 3).unwrap();
        crate::io_vtk::write_vtk_cells(&mut file, VtkElementType::TRIANGLE, &tri2vtx).unwrap();
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
