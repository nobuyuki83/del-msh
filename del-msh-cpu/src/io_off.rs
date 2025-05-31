//! methods for OFF files

#[allow(clippy::identity_op)]
pub fn save_tri_mesh<P: AsRef<std::path::Path>, T>(filepath: P, tri2vtx: &[usize], vtx2xyz: &[T])
where
    T: std::fmt::Display,
{
    let mut file = std::fs::File::create(filepath).expect("file not found.");
    use std::io::Write;
    let num_tri = tri2vtx.len() / 3;
    let num_vtx = vtx2xyz.len() / 3;
    writeln!(file, "OFF {} {} 0", num_vtx, num_tri).expect("fail");
    for i_vtx in 0..num_vtx {
        writeln!(
            file,
            "{} {} {}",
            vtx2xyz[i_vtx * 3 + 0],
            vtx2xyz[i_vtx * 3 + 1],
            vtx2xyz[i_vtx * 3 + 2]
        )
        .expect("fail");
    }
    for i_tri in 0..num_tri {
        writeln!(
            file,
            "3 {} {} {}",
            tri2vtx[i_tri * 3 + 0],
            tri2vtx[i_tri * 3 + 1],
            tri2vtx[i_tri * 3 + 2]
        )
        .expect("fail");
    }
}

/// load OFF file and output triangle mesh
/// * `file_path` - path to the file
pub fn load_as_tri_mesh<P: AsRef<std::path::Path>, Index, Real>(
    file_path: P,
) -> anyhow::Result<(Vec<Index>, Vec<Real>)>
where
    Real: std::str::FromStr,
    <Real as std::str::FromStr>::Err: std::fmt::Debug,
    Index: num_traits::PrimInt + 'static,
    usize: num_traits::AsPrimitive<Index>,
{
    let file = std::fs::File::open(file_path)?;
    let mut reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut line = String::new();
    let _ = reader.read_line(&mut line);
    let strs = line.clone();
    let strs: Vec<_> = strs.split_whitespace().collect();
    line.clear();
    assert_eq!(strs[0], "OFF");
    use std::str::FromStr;
    let num_vtx = usize::from_str(strs[1])?;
    let num_elem = usize::from_str(strs[2])?;
    // dbg!(num_vtx, num_elem);
    let mut vtx2xyz = Vec::<Real>::with_capacity(num_vtx * 3);
    for _i_vtx in 0..num_vtx {
        let _ = reader.read_line(&mut line);
        let strs = line.clone();
        let strs: Vec<_> = strs.split_whitespace().collect();
        line.clear();
        assert_eq!(strs.len(), 3);
        let x = Real::from_str(strs[0]).unwrap();
        let y = Real::from_str(strs[1]).unwrap();
        let z = Real::from_str(strs[2]).unwrap();
        vtx2xyz.push(x);
        vtx2xyz.push(y);
        vtx2xyz.push(z);
    }
    let mut elem2vtx = Vec::<Index>::with_capacity(num_elem * 3);
    for _i_elem in 0..num_elem {
        let _ = reader.read_line(&mut line);
        let strs = line.clone();
        let strs: Vec<_> = strs.split_whitespace().collect();
        line.clear();
        assert_eq!(strs.len(), 4);
        assert_eq!(strs[0], "3");
        let i0 = usize::from_str(strs[1]).unwrap();
        let i1 = usize::from_str(strs[2]).unwrap();
        let i2 = usize::from_str(strs[3]).unwrap();
        use num_traits::AsPrimitive;
        elem2vtx.push(i0.as_());
        elem2vtx.push(i1.as_());
        elem2vtx.push(i2.as_());
    }
    Ok((elem2vtx, vtx2xyz))
}
