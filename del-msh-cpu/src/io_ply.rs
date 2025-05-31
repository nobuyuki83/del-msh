//! methods for Ply file formats

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

pub trait XyzRgb {
    fn new(xyz: [f64; 3], rgb: [u8; 3]) -> Self;
}

pub fn read_xyzrgb<Path: AsRef<std::path::Path>, _XyzRgb: XyzRgb>(
    path: Path,
) -> anyhow::Result<Vec<_XyzRgb>> {
    // let file_path = "C:/Users/nobuy/Downloads/juice_box.ply";
    // let file_path = "/Users/nobuyuki/project/juice_box1.ply";
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut line = String::new();
    {
        // read magic number
        let _hoge = reader.read_line(&mut line)?;
        assert_eq!(line, "ply\n");
        line.clear();
    }
    {
        // read format
        let _hoge = reader.read_line(&mut line)?;
        let strs: Vec<_> = line.split_whitespace().collect();
        assert_eq!(strs[0], "format");
        let _format = match strs[1] {
            "binary_little_endian" => Format::BinaryLittleEndian,
            "binary_big_endian" => Format::BinaryBigEndian,
            "ascii" => Format::Ascii,
            &_ => panic!(),
        };
        line.clear();
    }
    {
        // read comment
        let _ = reader.read_line(&mut line)?;
        let strs: Vec<_> = line.split_whitespace().collect();
        assert_eq!(strs[0], "comment");
        line.clear();
    }
    let num_elem = {
        // read element number
        let _ = reader.read_line(&mut line)?; // element vertex
        let strs: Vec<_> = line.split_whitespace().collect();
        assert_eq!(strs[0], "element");
        use std::str::FromStr;
        usize::from_str(strs[2]).unwrap()
    };
    {
        // read_property
        let _ = reader.read_line(&mut line)?; // property double x
        let _ = reader.read_line(&mut line)?; // property double y
        let _ = reader.read_line(&mut line)?; // property double z
        let _ = reader.read_line(&mut line)?; // property uchar red
        let _ = reader.read_line(&mut line)?; // property uchar green
        let _ = reader.read_line(&mut line)?; // property uchar blue
        line.clear();
    }
    {
        // read "end header"
        let _ = reader.read_line(&mut line)?;
        assert_eq!(line, "end_header\n");
    }
    //
    let mut buf: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buf)?;
    let mut pnt2xyzrgb: Vec<_XyzRgb> = vec![];
    for i_elem in 0..num_elem {
        let i_byte: usize = i_elem * (8 * 3 + 3);
        let x = f64::from_le_bytes(buf[i_byte..i_byte + 8].try_into()?);
        let y = f64::from_le_bytes(buf[i_byte + 8..i_byte + 16].try_into()?);
        let z = f64::from_le_bytes(buf[i_byte + 16..i_byte + 24].try_into()?);
        let r = u8::from_le_bytes(buf[i_byte + 24..i_byte + 25].try_into()?);
        let g = u8::from_le_bytes(buf[i_byte + 25..i_byte + 26].try_into()?);
        let b = u8::from_le_bytes(buf[i_byte + 26..i_byte + 27].try_into()?);
        pnt2xyzrgb.push(XyzRgb::new([x, y, z], [r, g, b]));
    }
    Ok(pnt2xyzrgb)
}

fn parse_f32<const N: usize>(buff: &[u8], i_buff: usize) -> anyhow::Result<[f32; N]> {
    let mut a = [0f32; N];
    for i in 0..N {
        a[i] = f32::from_le_bytes(buff[i_buff + 4 * i..i_buff + 4 * i + 4].try_into()?);
    }
    Ok(a)
}

// ---------------------------
pub trait GaussSplat3D {
    fn new(
        xyz: [f32; 3],
        rgb_dc: [f32; 3],
        rgb_sh: [f32; 45],
        opacity: f32,
        scale: [f32; 3],
        quaternion: [f32; 4],
    ) -> Self;
}

pub fn read_3d_gauss_splat<Path: AsRef<std::path::Path>, Splat: GaussSplat3D>(
    path: Path,
) -> anyhow::Result<Vec<Splat>> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    use std::io::BufRead;
    let mut line = String::new();
    let _hoge = reader.read_line(&mut line)?;
    assert_eq!(line, "ply\n");
    line.clear();
    {
        // reading format
        let _hoge = reader.read_line(&mut line)?;
        let strs: Vec<_> = line.split_whitespace().collect();
        assert_eq!(strs[0], "format");
        let _format = match strs[1] {
            "binary_little_endian" => Format::BinaryLittleEndian,
            "binary_big_endian" => Format::BinaryBigEndian,
            "ascii" => Format::Ascii,
            &_ => panic!(),
        };
        line.clear();
    }
    {
        // reading comment
        let _ = reader.read_line(&mut line)?;
        let strs: Vec<_> = line.split_whitespace().collect();
        assert_eq!(strs[0], "comment");
        line.clear();
    }
    let num_elem = {
        let _ = reader.read_line(&mut line)?; // element vertex
        let strs: Vec<_> = line.split_whitespace().collect();
        assert_eq!(strs[0], "element");
        assert_eq!(strs[1], "vertex");
        use std::str::FromStr;
        let num_elem = usize::from_str(strs[2]).unwrap();
        line.clear();
        num_elem
    };
    //
    for _i in 0..62 {
        let _ = reader.read_line(&mut line)?; // property double x
        let strs: Vec<_> = line.split_whitespace().collect();
        //dbg!(&line);
        assert_eq!(strs[0], "property");
        assert_eq!(strs[1], "float");
        line.clear();
    }
    {
        // end header
        let _ = reader.read_line(&mut line)?;
        assert_eq!(line, "end_header\n");
    }
    // let sh_c0 = 0.28209479177387814;
    let sh_c0 = 0.5f32;
    let mut buf: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buf)?;
    assert_eq!(buf.len(), 62 * num_elem * 4);
    let mut pnt2gs3: Vec<Splat> = vec![];
    for i_elem in 0..num_elem {
        let xyz = parse_f32::<3>(&buf, i_elem * 62 * 4)?;
        let rgb = parse_f32::<3>(&buf, i_elem * 62 * 4 + 6 * 4)?;
        let sh = parse_f32::<45>(&buf, i_elem * 62 * 4 + 9 * 4)?;
        let op = parse_f32::<1>(&buf, i_elem * 62 * 4 + 54 * 4)?;
        let scale = parse_f32::<3>(&buf, i_elem * 62 * 4 + 55 * 4)?;
        let quaternion = parse_f32::<4>(&buf, i_elem * 62 * 4 + 58 * 4)?;
        //
        let rgb = [
            (rgb[0] + 0.5) * sh_c0,
            (rgb[1] + 0.5) * sh_c0,
            (rgb[2] + 0.5) * sh_c0,
        ];
        let quaternion = [quaternion[1], quaternion[2], quaternion[3], quaternion[0]];
        let quaternion = del_geo_core::quaternion::normalized(&quaternion);
        {
            // rotation in x direction to make the scene y-up?
            // let quat_x= del_geo_core::quaternion::from_axisangle(&[std::f32::consts::FRAC_PI_2,0.,0.]);
            // let quaternion = del_geo_core::quaternion::mult_quaternion(&quat_x, &quaternion);
            // let xyz = [xyz[0], -xyz[2], xyz[1]];
            // let quaternion = del_geo_core::quaternion::mult_quaternion(&quaternion, &quat_x);
        }
        let scale = [scale[0].exp(), scale[1].exp(), scale[2].exp()];
        let op = op[0];
        let op = 1f32 / (1f32 + (-op).exp());
        pnt2gs3.push(Splat::new(xyz, rgb, sh, op, scale, quaternion));
    }
    Ok(pnt2gs3)
}
