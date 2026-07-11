use anyhow::Context;
use std::io::BufRead;

pub fn read_elem<T: BufRead, IDX>(reader: &mut T, num_elem_cum: usize) -> Option<(usize, Vec<IDX>)>
where
    IDX: num_traits::PrimInt + 'static,
    usize: num_traits::AsPrimitive<IDX>,
{
    let mut buff = String::new();
    let (name, num_elem) = {
        // Peek without consuming: fill_buf() returns a reference into the
        // internal buffer; consume() advances the reader only if we proceed.
        let available = reader.fill_buf().ok()?;
        let line_len = available
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| p + 1)
            .unwrap_or(available.len());
        let peeked = std::str::from_utf8(&available[..line_len]).ok()?;
        let a: Vec<&str> = peeked.split_whitespace().collect();
        if a.len() != 6 || a[0..3] != ["#", "number", "of"] {
            return None; // reader not advanced — caller can try another section
        }
        let name = a[3].to_owned();
        let num_elem = a[5]
            .parse::<usize>()
            .context("failed to parse number of elements")
            .ok()?;
        reader.consume(line_len); // actually advance past the header line
        (name, num_elem)
    };
    let num_node = match name.as_str() {
        "tetra" => 4,
        "pyramid" => 5,
        "prism" => 6,
        "hexahedra" => 8,
        _ => {
            unreachable!()
        }
    };
    let mut elem2vtx = vec![IDX::zero(); num_elem * num_node];
    for i_elem in 0..num_elem {
        buff.clear();
        reader
            .read_line(&mut buff)
            .expect("failed to read element line");
        let a: Vec<&str> = buff.split_whitespace().collect();
        assert_eq!(
            a.len(),
            num_node + 1,
            "unexpected element line length: {}",
            buff.trim()
        );
        let idx = a[0]
            .parse::<usize>()
            .expect("failed to parse element index");
        assert_eq!(
            idx,
            num_elem_cum + i_elem + 1,
            "element index mismatch: {} != {}",
            idx,
            num_elem_cum + i_elem + 1
        );
        for i_node in 0..num_node {
            let tmp = a[i_node + 1]
                .parse::<usize>()
                .expect("failed to parse vertex index");
            assert!(tmp >= 1, "vertex index must be >= 1, got {}", tmp);
            use num_traits::AsPrimitive;
            elem2vtx[i_elem * num_node + i_node] = (tmp - 1).as_();
        }
    }
    Some((num_node, elem2vtx))
}

pub fn read_value<T: BufRead>(reader: &mut T, num_vtx: usize) -> Option<(usize, Vec<f32>)> {
    let mut buff = String::new();
    let name = {
        // Peek without consuming: fill_buf() returns a reference into the
        // internal buffer; consume() advances the reader only if we proceed.
        let available = reader.fill_buf().ok()?;
        let line_len = available
            .iter()
            .position(|&b| b == b'\n')
            .map(|p| p + 1)
            .unwrap_or(available.len());
        let peeked = std::str::from_utf8(&available[..line_len]).ok()?;
        let a: Vec<&str> = peeked.split_whitespace().collect();
        if a.len() != 2 || a[0] != "#" {
            return None; // reader not advanced — caller can try another section
        }
        let name = a[1].to_owned();
        reader.consume(line_len); // actually advance past the header line
        name
    };
    let num_node = match name.as_str() {
        "velocity" => 3,
        "pressure" => 1,
        _ => {
            unreachable!()
        }
    };
    let mut vtx2value = vec![0f32; num_vtx * num_node];
    for i_vtx in 0..num_vtx {
        buff.clear();
        reader
            .read_line(&mut buff)
            .expect("failed to read element line");
        let a: Vec<&str> = buff.split_whitespace().collect();
        assert_eq!(
            a.len(),
            num_node + 1,
            "unexpected element line length: {}",
            buff.trim()
        );
        let idx = a[0]
            .parse::<usize>()
            .expect("failed to parse element index");
        assert_eq!(
            idx,
            i_vtx + 1,
            "element index mismatch: {} != {}",
            idx,
            i_vtx + 1
        );
        for i_node in 0..num_node {
            let tmp = a[i_node + 1]
                .parse::<f32>()
                .expect("failed to parse vertex index");
            use num_traits::AsPrimitive;
            vtx2value[i_vtx * num_node + i_node] = tmp.as_();
        }
    }
    Some((num_node, vtx2value))
}

pub struct DataFromCfdMeshTxt<IDX> {
    pub vtx2xyz: Vec<f32>,
    pub tet2vtx: Vec<IDX>,
    pub pyrmd2vtx: Vec<IDX>,
    pub prism2vtx: Vec<IDX>,
    pub hex2vtx: Vec<IDX>,
    pub vtx2velo: Vec<f32>,
    pub vtx2pressure: Vec<f32>,
}

pub fn read<P: AsRef<std::path::Path>, IDX>(path: P) -> anyhow::Result<DataFromCfdMeshTxt<IDX>>
where
    IDX: num_traits::PrimInt + 'static,
    usize: num_traits::AsPrimitive<IDX>,
{
    let file = std::fs::File::open(&path)
        .with_context(|| format!("file not found: {:?}", path.as_ref()))?;
    let mut reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut buff = String::new();
    let num_vtx = {
        reader
            .read_line(&mut buff)
            .context("failed to read node count header")?;
        let a: Vec<&str> = buff.split_whitespace().collect();
        anyhow::ensure!(
            a[0..4] == ["#", "number", "of", "nodes"],
            "unexpected header: {}",
            buff.trim()
        );
        a[4].parse::<usize>()
            .context("failed to parse number of nodes")?
    };
    let vtx2xyz = {
        let mut node2xyz = vec![0f32; num_vtx * 3];
        for i_node in 0..num_vtx {
            buff.clear();
            reader
                .read_line(&mut buff)
                .context("failed to read node line")?;
            let a: Vec<&str> = buff.split_whitespace().collect();
            let idx = a[0]
                .parse::<usize>()
                .context("failed to parse node index")?;
            anyhow::ensure!(
                idx == i_node + 1,
                "node index mismatch: {} != {}",
                idx,
                i_node + 1
            );
            node2xyz[i_node * 3] = a[1].parse::<f32>().context("failed to parse x")?;
            node2xyz[i_node * 3 + 1] = a[2].parse::<f32>().context("failed to parse y")?;
            node2xyz[i_node * 3 + 2] = a[3].parse::<f32>().context("failed to parse z")?;
        }
        node2xyz
    };
    let _num_elem = {
        buff.clear();
        reader
            .read_line(&mut buff)
            .context("failed to read element count header")?;
        let a: Vec<&str> = buff.split_whitespace().collect();
        anyhow::ensure!(
            a[0..4] == ["#", "number", "of", "elements"],
            "unexpected header: {}",
            buff.trim()
        );
        a[4].parse::<usize>()
            .context("failed to parse number of elements")?
    };
    let mut tet2vtx = vec![];
    let mut prism2vtx = vec![];
    let mut pyrmd2vtx = vec![];
    let mut hex2vtx = vec![];
    let mut num_elem_cum = 0;
    while let Some((num_node, elem2vtx)) = read_elem::<_, IDX>(&mut reader, num_elem_cum) {
        num_elem_cum += elem2vtx.len() / num_node;
        match num_node {
            4 => {
                tet2vtx = elem2vtx;
            }
            5 => {
                pyrmd2vtx = elem2vtx;
            }
            6 => {
                prism2vtx = elem2vtx;
            }
            8 => {
                hex2vtx = elem2vtx;
            }
            _ => {
                unreachable!();
            }
        }
    }
    let mut vtx2velo: Vec<f32> = vec![];
    let mut vtx2pressure: Vec<f32> = vec![];
    while let Some((num_vdim, vtx2value)) = read_value::<_>(&mut reader, num_vtx) {
        match num_vdim {
            1 => {
                vtx2pressure = vtx2value;
            }
            3 => {
                vtx2velo = vtx2value;
            }
            _ => {
                unreachable!();
            }
        }
    }
    Ok(DataFromCfdMeshTxt {
        vtx2xyz,
        tet2vtx,
        pyrmd2vtx,
        prism2vtx,
        hex2vtx,
        vtx2velo,
        vtx2pressure,
    })
}

#[cfg(test)]
mod tests {

    fn check(data: &crate::io_cfd_mesh_txt::DataFromCfdMeshTxt<u32>, name: &str) {
        {
            let mut file =
                std::fs::File::create(format!("../target/{}.vtk", name)).expect("file not found.");
            crate::io_vtk::write_vtk_points(&mut file, "hoge", &data.vtx2xyz, 3).unwrap();
            crate::io_vtk::write_vtk_cells_mix(
                &mut file,
                &data.tet2vtx,
                &data.pyrmd2vtx,
                &data.prism2vtx,
                &data.hex2vtx,
            )
            .unwrap();
        }
        let (elem2idx_offset, idx2vtx) = {
            let num_tet = data.tet2vtx.len() / 4;
            let num_pyrmd = data.pyrmd2vtx.len() / 5;
            let num_prism = data.prism2vtx.len() / 6;
            let num_hex = data.hex2vtx.len() / 8;
            let mut elem2idx_offset = vec![0u32; num_tet + num_pyrmd + num_prism + num_hex + 1];
            let mut idx2vtx = vec![0u32; num_tet * 4 + num_pyrmd * 5 + num_prism * 6 + num_hex * 8];
            crate::mixed_mesh::to_polyhedron_mesh(
                &data.tet2vtx,
                &data.pyrmd2vtx,
                &data.prism2vtx,
                &data.hex2vtx,
                &mut elem2idx_offset,
                &mut idx2vtx,
            );
            (elem2idx_offset, idx2vtx)
        };
        {
            let num_elem = data.tet2vtx.len() / 4
                + data.pyrmd2vtx.len() / 5
                + data.prism2vtx.len() / 6
                + data.hex2vtx.len() / 8;
            let mut elem2volume = vec![0f32; num_elem];
            crate::polyhedron_mesh::elem2volume(
                &elem2idx_offset,
                &idx2vtx,
                &data.vtx2xyz,
                1,
                &mut elem2volume,
            );
            assert!((elem2volume[0] - 1. / 12.0).abs() < 1.0e-7);
            assert!((elem2volume[1] - 1. / 6.0).abs() < 1.0e-7);
            assert!((elem2volume[2] - 1. / 2.0).abs() < 1.0e-7);
        }
        let elem2cog = crate::elem2center::from_polygon_mesh_as_points(
            &elem2idx_offset,
            &idx2vtx,
            &data.vtx2xyz,
            3,
        );
        let bvhnodes = crate::bvhnodes_morton::from_vtx2xyz::<u32>(&elem2cog, 3);
        crate::bvhnodes::check_bvh_topology(&bvhnodes, elem2idx_offset.len() - 1);
        let _bvhnode2aabb = crate::bvhnode2aabb3::from_polygon_polyhedron_mesh_with_bvh(
            0,
            &bvhnodes,
            &elem2idx_offset,
            &idx2vtx,
            &data.vtx2xyz,
        );
    }

    #[test]
    fn test0() {
        {
            let name = "cfd_mesh";
            let data =
                crate::io_cfd_mesh_txt::read::<_, u32>(format!("../asset/{}.txt", name)).unwrap();
            assert_eq!(data.tet2vtx.len(), 4);
            assert_eq!(data.pyrmd2vtx.len(), 5);
            assert_eq!(data.prism2vtx.len(), 6);
            assert_eq!(data.hex2vtx.len(), 8);
            check(&data, name);
        }
        {
            let name = "cfd_mesh1";
            let data =
                crate::io_cfd_mesh_txt::read::<_, u32>(format!("../asset/{}.txt", name)).unwrap();
            assert_eq!(data.tet2vtx.len(), 4);
            assert_eq!(data.pyrmd2vtx.len(), 5);
            assert_eq!(data.prism2vtx.len(), 6);
            let num_vtx = data.vtx2xyz.len() / 3;
            assert_eq!(data.vtx2velo.len(), num_vtx * 3);
            assert_eq!(data.vtx2pressure.len(), num_vtx);
            check(&data, name);
        }
    }
}
