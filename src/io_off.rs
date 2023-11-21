//! methods for OFF files

pub fn save_tri_mesh<P: AsRef<std::path::Path>, T>(
    filepath: P,
    tri2vtx: &[usize],
    vtx2xyz: &[T])
where T: std::fmt::Display
{
    let mut file = std::fs::File::create(filepath).expect("file not found.");
    use std::io::Write;
    let num_tri = tri2vtx.len() / 3;
    let num_vtx  = vtx2xyz.len() / 3;
    writeln!(file, "OFF {} {} 0",
             num_vtx, num_tri).expect("fail");
    for i_vtx in 0..num_vtx {
        writeln!(file, "{} {} {}",
                 vtx2xyz[i_vtx * 3 + 0],
                 vtx2xyz[i_vtx * 3 + 1],
                 vtx2xyz[i_vtx * 3 + 2]).expect("fail");
    }
    for i_tri in 0..num_tri {
        writeln!(file, "3 {} {} {}",
                 tri2vtx[i_tri * 3 + 0],
                 tri2vtx[i_tri * 3 + 1],
                 tri2vtx[i_tri * 3 + 2]).expect("fail");
    }
}

pub fn load_as_tri_mesh<P: AsRef<std::path::Path>>(
    file_path: P) -> (Vec<usize>, Vec<f64>)
{
    let file = std::fs::File::open(file_path).expect("file not found.");
    let mut reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut line = String::new();
    let _ = reader.read_line(&mut line);
    let strs = line.clone();
    let strs: Vec<_> = strs.split_whitespace().collect();
    line.clear();
    assert_eq!(strs[0], "OFF");
    use std::str::FromStr;
    let num_vtx = usize::from_str(strs[1]).unwrap();
    let num_elem = usize::from_str(strs[2]).unwrap();
    dbg!(num_vtx, num_elem);
    let mut vtx2xyz = Vec::<f64>::new();
    vtx2xyz.reserve(num_vtx*3);
    for _i_vtx in 0..num_vtx {
        let _ = reader.read_line(&mut line);
        let strs = line.clone();
        let strs: Vec<_> = strs.split_whitespace().collect();
        line.clear();
        assert_eq!(strs.len(),3);
        let x = f64::from_str(strs[0]).unwrap();
        let y = f64::from_str(strs[1]).unwrap();
        let z = f64::from_str(strs[2]).unwrap();
        vtx2xyz.push(x);
        vtx2xyz.push(y);
        vtx2xyz.push(z);
    }
    let mut elem2vtx = Vec::<usize>::new();
    elem2vtx.reserve(num_elem*3);
    for _i_elem in 0..num_elem {
        let _ = reader.read_line(&mut line);
        let strs = line.clone();
        let strs: Vec<_> = strs.split_whitespace().collect();
        line.clear();
        assert_eq!(strs.len(),4);
        assert_eq!(strs[0], "3");
        let i0 = usize::from_str(strs[1]).unwrap();
        let i1 = usize::from_str(strs[2]).unwrap();
        let i2 = usize::from_str(strs[3]).unwrap();
        elem2vtx.push(i0);
        elem2vtx.push(i1);
        elem2vtx.push(i2);
    }
    (elem2vtx, vtx2xyz)
}