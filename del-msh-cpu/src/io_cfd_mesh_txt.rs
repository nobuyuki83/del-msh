use anyhow::Context;
use std::io::BufRead;

pub fn read_elem<T: BufRead, IDX>(
    reader: &mut T,
    name: &str,
    num_node: usize,
    num_elem_cum: usize,
) -> anyhow::Result<Vec<IDX>>
where
    IDX: num_traits::PrimInt + 'static,
    usize: num_traits::AsPrimitive<IDX>,
{
    let mut buff = String::new();
    let num_elem = {
        buff.clear();
        reader
            .read_line(&mut buff)
            .context("failed to read element count header")?;
        let a: Vec<&str> = buff.split_whitespace().collect();
        anyhow::ensure!(
            a[0..5] == ["#", "number", "of", name, "elements"],
            "unexpected header: {}",
            buff.trim()
        );
        a[5].parse::<usize>()
            .context("failed to parse number of elements")?
    };
    let mut elem2vtx = vec![IDX::zero(); num_elem * num_node];
    for i_elem in 0..num_elem {
        buff.clear();
        reader
            .read_line(&mut buff)
            .context("failed to read element line")?;
        let a: Vec<&str> = buff.split_whitespace().collect();
        anyhow::ensure!(
            a.len() == num_node + 1,
            "unexpected element line length: {}",
            buff.trim()
        );
        let idx = a[0]
            .parse::<usize>()
            .context("failed to parse element index")?;
        anyhow::ensure!(
            idx == num_elem_cum + i_elem + 1,
            "element index mismatch: {} != {}",
            idx,
            num_elem_cum + i_elem + 1
        );
        for i_node in 0..num_node {
            let tmp = a[i_node + 1]
                .parse::<usize>()
                .context("failed to parse vertex index")?;
            anyhow::ensure!(tmp >= 1, "vertex index must be >= 1, got {}", tmp);
            use num_traits::AsPrimitive;
            elem2vtx[i_elem * num_node + i_node] = (tmp - 1).as_();
        }
    }
    Ok(elem2vtx)
}

pub struct DataFromCfdMeshTxt<IDX> {
    pub vtx2xyz: Vec<f32>,
    pub tet2vtx: Vec<IDX>,
    pub pyrmd2vtx: Vec<IDX>,
    pub prism2vtx: Vec<IDX>,
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
    let num_elem = {
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
    let tet2vtx = read_elem::<_, IDX>(&mut reader, "tetra", 4, 0)?;
    let num_tet = tet2vtx.len() / 4;
    let pyrmd2vtx = read_elem::<_, IDX>(&mut reader, "pyramid", 5, num_tet)?;
    let num_pyrmd = pyrmd2vtx.len() / 5;
    let prism2vtx = read_elem::<_, IDX>(&mut reader, "prism", 6, num_tet + num_pyrmd)?;
    let num_prism = prism2vtx.len() / 6;
    anyhow::ensure!(
        num_elem == num_tet + num_pyrmd + num_prism,
        "element count mismatch: expected {}, got {}",
        num_elem,
        num_tet + num_pyrmd + num_prism
    );
    Ok(DataFromCfdMeshTxt {
        vtx2xyz,
        tet2vtx,
        pyrmd2vtx,
        prism2vtx,
    })
}

#[test]
fn hoge() {
    let data = read::<_, u32>("../asset/cfd_mesh.txt").unwrap();
    let mut file = std::fs::File::create("../target/cfd_mesh.vtk").expect("file not found.");
    crate::io_vtk::write_vtk_points(&mut file, "hoge", &data.vtx2xyz, 3).unwrap();
    assert_eq!(data.tet2vtx.len(), 4);
    assert_eq!(data.pyrmd2vtx.len(), 5);
    assert_eq!(data.prism2vtx.len(), 6);
    crate::io_vtk::write_vtk_cells_mix(&mut file, &data.tet2vtx, &data.pyrmd2vtx, &data.prism2vtx)
        .unwrap();
    let (elem2idx_offset, idx2vtx) = {
        let num_tet = data.tet2vtx.len() / 4;
        let num_pyrmd = data.pyrmd2vtx.len() / 5;
        let num_prism = data.prism2vtx.len() / 6;
        let mut elem2idx_offset = vec![0u32; num_tet + num_pyrmd + num_prism + 1];
        let mut idx2vtx = vec![0u32; num_tet * 4 + num_pyrmd * 5 + num_prism * 6];
        crate::mixed_mesh::to_polyhedron_mesh(
            &data.tet2vtx,
            &data.pyrmd2vtx,
            &data.prism2vtx,
            &mut elem2idx_offset,
            &mut idx2vtx,
        );
        (elem2idx_offset, idx2vtx)
    };
    {
        let num_elem = data.tet2vtx.len() / 4 + data.pyrmd2vtx.len() / 5 + data.prism2vtx.len() / 6;
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
    let bvhnode2aabb = crate::bvhnode2aabb3::from_polygon_polyhedron_mesh_with_bvh(
        0,
        &bvhnodes,
        &elem2idx_offset,
        &idx2vtx,
        &data.vtx2xyz,
    );
}
