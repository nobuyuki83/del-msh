use std::io::Read;
enum Format {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

/* --------------------------------------*/
// below: XyzRgb

/*
pub struct XyzRgb {
    pub xyz: [f64; 3],
    pub rgb: [u8; 3],
}

impl crate::vtx2xyz::HasXyz<f64> for XyzRgb {
    fn xyz(&self) -> [f64; 3] {
        self.xyz
    }
}
 */

pub trait Xyz {
    fn set_xyz(&mut self, xyz: &[f64;3]);
}

pub trait Rgb {
    fn set_rgb(&mut self, rgb: &[u8;3]);
}



pub fn read_xyzrgb<Path: AsRef<std::path::Path>, XyzRgb: Xyz + Rgb + Default>(path: Path) -> anyhow::Result<Vec<XyzRgb>> {
    // let file_path = "C:/Users/nobuy/Downloads/juice_box.ply";
    // let file_path = "/Users/nobuyuki/project/juice_box1.ply";
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut line = String::new();
    let _hoge = reader.read_line(&mut line)?;
    assert_eq!(line, "ply\n");
    line.clear();
    let _hoge = reader.read_line(&mut line)?;
    dbg!(&line);
    let strs: Vec<_> = line.split_whitespace().collect();
    assert_eq!(strs[0], "format");
    let _format = match strs[1] {
        "binary_little_endian" => Format::BinaryLittleEndian,
        "binary_big_endian" => Format::BinaryBigEndian,
        "ascii" => Format::Ascii,
        &_ => panic!(),
    };
    let _ = reader.read_line(&mut line)?;
    dbg!(&line);
    line.clear();
    //
    let _ = reader.read_line(&mut line)?; // element vertex
    dbg!(&line);
    let strs: Vec<_> = line.split_whitespace().collect();
    assert_eq!(strs[0], "element");
    use std::str::FromStr;
    let num_elem = usize::from_str(strs[2]).unwrap();
    dbg!(num_elem);
    //
    let _ = reader.read_line(&mut line)?; // property double x
    let _ = reader.read_line(&mut line)?; // property double y
    let _ = reader.read_line(&mut line)?; // property double z
    let _ = reader.read_line(&mut line)?; // property uchar red
    let _ = reader.read_line(&mut line)?; // property uchar green
    let _ = reader.read_line(&mut line)?; // property uchar blue
    line.clear();
    //
    let _ = reader.read_line(&mut line)?; // end_header
    assert_eq!(line, "end_header\n");
    //
    let mut buf: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buf)?;
    let mut pnt2xyzrgb: Vec<XyzRgb> = vec![];
    for i_elem in 0..num_elem {
        let i_byte: usize = i_elem * (8 * 3 + 3);
        let x = f64::from_le_bytes(buf[i_byte..i_byte + 8].try_into()?);
        let y = f64::from_le_bytes(buf[i_byte + 8..i_byte + 16].try_into()?);
        let z = f64::from_le_bytes(buf[i_byte + 16..i_byte + 24].try_into()?);
        let r = u8::from_le_bytes(buf[i_byte + 24..i_byte + 25].try_into()?);
        let g = u8::from_le_bytes(buf[i_byte + 25..i_byte + 26].try_into()?);
        let b = u8::from_le_bytes(buf[i_byte + 26..i_byte + 27].try_into()?);
        // dbg!((i_byte, i_elem, i_elem * (8 * 3 + 3), x,y,z));
        let mut pnt = XyzRgb::default();
        pnt.set_xyz(&[x,y,z]);
        pnt.set_rgb(&[r,g,b]);
        pnt2xyzrgb.push(pnt);
    }
    Ok(pnt2xyzrgb)
}
