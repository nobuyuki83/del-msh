//! reading Nastran file
use num_traits::AsPrimitive;

fn parse(s: &str) -> f32 {
    let s = s.trim_start();
    if let Some(i) = s.rfind('-') {
        if i != 0 {
            let mut s = String::from(s);
            s.insert(i, 'e');
            let v = s.trim().parse::<f32>().unwrap();
            //  dbg!(i,&s,&v);
            return v;
        }
    } else if let Some(i) = s.rfind('+') {
        assert_ne!(i, 0);
        let mut s = String::from(s);
        s.insert(i, 'e');
        let v = s.trim().parse::<f32>().unwrap();
        //  dbg!(i,&s,&v);
        return v;
    }
    s.trim().parse::<f32>().ok().unwrap()
}

pub fn load_tri_mesh<P, Index>(path: P) -> (Vec<Index>, Vec<f32>)
where
    P: AsRef<std::path::Path>,
    Index: num_traits::PrimInt + std::str::FromStr + 'static + AsPrimitive<usize>,
    <Index as std::str::FromStr>::Err: std::fmt::Debug,
    usize: AsPrimitive<Index>,
{
    let mut vtx2xyz = vec![0f32; 0];
    let mut vtx2idx: Vec<Index> = vec![];
    let mut tri2idx: Vec<Index> = vec![];
    //
    let file = std::fs::File::open(path).expect("file not found.");
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    for line in reader.lines().map_while(Result::ok) {
        // println!("{:?}", line);
        if line.starts_with("GRID") {
            assert_eq!(line.len(), 48);
            let _ = line.get(0..9);
            let idx = line.get(9..24);
            let idx = idx.unwrap().trim().parse::<Index>().unwrap();
            let v0 = line.get(24..32).unwrap();
            let v1 = line.get(32..40).unwrap();
            let v2 = line.get(40..48).unwrap();
            // dbg!(v0, v1, v2);
            let v0 = parse(v0);
            let v1 = parse(v1);
            let v2 = parse(v2);
            vtx2xyz.push(v0);
            vtx2xyz.push(v1);
            vtx2xyz.push(v2);
            vtx2idx.push(idx);
        } else if line.starts_with("CTRIA3") {
            assert_eq!(line.len(), 48);
            let _ = line.get(0..8); // CTRIA3
            let _ = line.get(8..16);
            let _ = line.get(16..24); // I don't know what this number is for
            let v0 = line.get(24..32).unwrap().trim();
            let v1 = line.get(32..40).unwrap().trim();
            let v2 = line.get(40..48).unwrap().trim();
            let v0 = v0.parse::<Index>().unwrap();
            let v1 = v1.parse::<Index>().unwrap();
            let v2 = v2.parse::<Index>().unwrap();
            tri2idx.push(v0);
            tri2idx.push(v1);
            tri2idx.push(v2);
        }
    }
    //
    let &num_idx = vtx2idx.iter().max().unwrap();
    let mut idx2vtx = vec![Index::max_value(); num_idx.as_() + 1];
    vtx2idx
        .iter()
        .enumerate()
        .for_each(|(vtx, &idx)| idx2vtx[idx.as_()] = vtx.as_());
    let mut tri2vtx = vec![Index::zero(); tri2idx.len()];
    tri2vtx
        .iter_mut()
        .zip(tri2idx)
        .for_each(|(vtx, idx)| *vtx = idx2vtx[idx.as_()]);
    (tri2vtx, vtx2xyz)
}
