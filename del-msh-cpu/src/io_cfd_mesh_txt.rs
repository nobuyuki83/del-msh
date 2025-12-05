use std::io::BufRead;

pub fn read_elem<T: BufRead>(
    reader: &mut T,
    name: &str,
    num_node: usize,
    num_elem_cum: usize,
) -> Vec<usize> {
    let mut buff = String::new();
    let num_elem = {
        buff.clear();
        reader.read_line(&mut buff).unwrap();
        let a: Vec<&str> = buff.split_whitespace().collect();
        assert_eq!(a[0..5], ["#", "number", "of", name, "elements"]);
        a[5].parse::<usize>().unwrap()
    };
    let mut elem2vtx = vec![0usize; num_elem * num_node];
    for i_elem in 0..num_elem {
        buff.clear();
        reader.read_line(&mut buff).unwrap();
        let a: Vec<&str> = buff.split_whitespace().collect();
        assert_eq!(a.len(), num_node + 1);
        assert_eq!(
            a[0].parse::<usize>().unwrap(),
            num_elem_cum + i_elem + 1,
            "{} {}",
            i_elem,
            &a[0]
        );
        for i_node in 0..num_node {
            let tmp = a[i_node + 1].parse::<usize>().unwrap();
            assert!(tmp >= 1);
            elem2vtx[i_elem * num_node + i_node] = tmp - 1;
        }
    }
    elem2vtx
}

pub struct MixedMeshForCfd {
    pub vtx2xyz: Vec<f32>,
    pub tet2vtx: Vec<usize>,
    pub pyrmd2vtx: Vec<usize>,
    pub prism2vtx: Vec<usize>,
}

pub fn read<P: AsRef<std::path::Path>>(path: P) -> MixedMeshForCfd {
    let file = std::fs::File::open(path).expect("file not found.");
    let mut reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut buff = String::new();
    let num_vtx = {
        reader.read_line(&mut buff).unwrap();
        let a: Vec<&str> = buff.split_whitespace().collect();
        assert_eq!(a[0..4], ["#", "number", "of", "nodes"]);
        a[4].parse::<usize>().unwrap()
    };
    let vtx2xyz = {
        let mut node2xyz = vec![0f32; num_vtx * 3];
        for i_node in 0..num_vtx {
            buff.clear();
            reader.read_line(&mut buff).unwrap();
            let a: Vec<&str> = buff.split_whitespace().collect();
            assert_eq!(
                a[0].parse::<usize>().unwrap(),
                i_node + 1,
                "{} {}",
                i_node,
                &a[0]
            );
            node2xyz[i_node * 3] = a[1].parse::<f32>().unwrap();
            node2xyz[i_node * 3 + 1] = a[2].parse::<f32>().unwrap();
            node2xyz[i_node * 3 + 2] = a[3].parse::<f32>().unwrap();
        }
        node2xyz
    };
    let num_elem = {
        buff.clear();
        reader.read_line(&mut buff).unwrap();
        let a: Vec<&str> = buff.split_whitespace().collect();
        assert_eq!(a[0..4], ["#", "number", "of", "elements"]);
        a[4].parse::<usize>().unwrap()
    };
    let tet2vtx = read_elem(&mut reader, "tetra", 4, 0);
    let num_tet = tet2vtx.len() / 4;
    let pyrmd2vtx = read_elem(&mut reader, "pyramid", 5, num_tet);
    let num_pyrmd = pyrmd2vtx.len() / 5;
    let prism2vtx = read_elem(&mut reader, "prism", 6, num_tet + num_pyrmd);
    let num_prism = prism2vtx.len() / 6;
    assert_eq!(num_elem, num_tet + num_pyrmd + num_prism);
    MixedMeshForCfd {
        vtx2xyz,
        tet2vtx,
        pyrmd2vtx,
        prism2vtx,
    }
}

#[test]
fn hoge() {
    let data = read("../asset/cfd_mesh.txt");
    let mut file = std::fs::File::create("../target/cfd_mesh.vtk").expect("file not found.");
    crate::io_vtk::write_vtk_points(&mut file, "hoge", &data.vtx2xyz, 3).unwrap();
    dbg!(data.tet2vtx.len() / 4);
    dbg!(data.pyrmd2vtx.len() / 5);
    dbg!(data.prism2vtx.len() / 6);
    crate::io_vtk::write_vtk_cells_mix(&mut file, &data.tet2vtx, &data.pyrmd2vtx, &data.prism2vtx)
        .unwrap();
}
