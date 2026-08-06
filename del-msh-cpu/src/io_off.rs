//! methods for OFF files

#[allow(clippy::identity_op)]
pub fn save_tri_mesh<P: AsRef<std::path::Path>, T>(
    filepath: P,
    tri2vtx: &[[usize; 3]],
    vtx2xyz: &[[T; 3]],
) where
    T: std::fmt::Display,
{
    let mut file = std::fs::File::create(filepath).expect("file not found.");
    use std::io::Write;
    let num_tri = tri2vtx.len();
    let num_vtx = vtx2xyz.len();
    writeln!(file, "OFF {num_vtx} {num_tri} 0").expect("fail");
    for p in vtx2xyz.iter() {
        writeln!(file, "{} {} {}", p[0], p[1], p[2]).expect("fail");
    }
    for t in tri2vtx.iter() {
        writeln!(file, "3 {} {} {}", t[0], t[1], t[2]).expect("fail");
    }
}

/// load OFF file and output triangle mesh
/// * `file_path` - path to the file
#[allow(clippy::type_complexity)]
pub fn load_as_tri_mesh<P: AsRef<std::path::Path>, Index, Real>(
    file_path: P,
) -> anyhow::Result<(Vec<[Index; 3]>, Vec<[Real; 3]>)>
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
    let mut vtx2xyz = Vec::<[Real; 3]>::with_capacity(num_vtx);
    for _i_vtx in 0..num_vtx {
        let _ = reader.read_line(&mut line);
        let strs = line.clone();
        let strs: Vec<_> = strs.split_whitespace().collect();
        line.clear();
        assert_eq!(strs.len(), 3);
        let x = Real::from_str(strs[0]).unwrap();
        let y = Real::from_str(strs[1]).unwrap();
        let z = Real::from_str(strs[2]).unwrap();
        vtx2xyz.push([x, y, z]);
    }
    let mut elem2vtx = Vec::<[Index; 3]>::with_capacity(num_elem);
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
        elem2vtx.push([i0.as_(), i1.as_(), i2.as_()]);
    }
    Ok((elem2vtx, vtx2xyz))
}
