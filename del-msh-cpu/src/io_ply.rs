use anyhow::Context;

// ---- private PLY header types ----

#[derive(Clone)]
enum PlyTy {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyTy {
    fn from_str(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => anyhow::bail!("unknown PLY scalar type: {s}"),
        })
    }
    fn size(&self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
    fn read_f64(&self, buf: &[u8], le: bool) -> f64 {
        macro_rules! prim {
            ($t:ty, $n:expr) => {{
                let b: [u8; $n] = buf[..$n].try_into().unwrap();
                (if le {
                    <$t>::from_le_bytes(b)
                } else {
                    <$t>::from_be_bytes(b)
                }) as f64
            }};
        }
        match self {
            Self::I8 => buf[0] as i8 as f64,
            Self::U8 => buf[0] as f64,
            Self::I16 => prim!(i16, 2),
            Self::U16 => prim!(u16, 2),
            Self::I32 => prim!(i32, 4),
            Self::U32 => prim!(u32, 4),
            Self::F32 => prim!(f32, 4),
            Self::F64 => prim!(f64, 8),
        }
    }
    fn parse_ascii(&self, s: &str) -> anyhow::Result<f64> {
        Ok(match self {
            Self::I8 => s.parse::<i8>()? as f64,
            Self::U8 => s.parse::<u8>()? as f64,
            Self::I16 => s.parse::<i16>()? as f64,
            Self::U16 => s.parse::<u16>()? as f64,
            Self::I32 => s.parse::<i32>()? as f64,
            Self::U32 => s.parse::<u32>()? as f64,
            Self::F32 => s.parse::<f32>()? as f64,
            Self::F64 => s.parse::<f64>()?,
        })
    }
}

#[derive(Clone)]
enum PlyProp {
    Scalar {
        name: String,
        ty: PlyTy,
    },
    List {
        name: String,
        cnt_ty: PlyTy,
        val_ty: PlyTy,
    },
}

struct PlyElem {
    name: String,
    count: usize,
    props: Vec<PlyProp>,
}

// ---- public API ----

pub fn load_as_tri_mesh<P: AsRef<std::path::Path>, Index, Real>(
    file_path: P,
) -> anyhow::Result<crate::trimesh3::TriMesh3<Index, Real>>
where
    Index: num_traits::PrimInt + 'static,
    usize: num_traits::AsPrimitive<Index>,
    Real: num_traits::Float + 'static,
    f64: num_traits::AsPrimitive<Real>,
{
    use num_traits::AsPrimitive;
    use std::io::{BufRead, Read};

    // ---- parse header ----
    let file = std::fs::File::open(file_path.as_ref())
        .with_context(|| format!("cannot open {:?}", file_path.as_ref()))?;
    let mut rdr = std::io::BufReader::new(file);
    let mut line = String::new();

    rdr.read_line(&mut line)?;
    anyhow::ensure!(line.trim() == "ply", "not a PLY file");

    let mut is_ascii = false;
    let mut is_le = true;
    let mut elems: Vec<PlyElem> = vec![];
    let mut cur: Option<PlyElem> = None;

    loop {
        line.clear();
        rdr.read_line(&mut line)?;
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        match t[0] {
            "end_header" => {
                if let Some(e) = cur.take() {
                    elems.push(e);
                }
                break;
            }
            "format" => {
                is_ascii = t[1] == "ascii";
                is_le = t[1] != "binary_big_endian";
            }
            "element" => {
                if let Some(e) = cur.take() {
                    elems.push(e);
                }
                cur = Some(PlyElem {
                    name: t[1].to_owned(),
                    count: t[2].parse()?,
                    props: vec![],
                });
            }
            "property" => {
                if let Some(e) = cur.as_mut() {
                    if t[1] == "list" {
                        e.props.push(PlyProp::List {
                            name: t[4].to_owned(),
                            cnt_ty: PlyTy::from_str(t[2])?,
                            val_ty: PlyTy::from_str(t[3])?,
                        });
                    } else {
                        e.props.push(PlyProp::Scalar {
                            name: t[2].to_owned(),
                            ty: PlyTy::from_str(t[1])?,
                        });
                    }
                }
            }
            _ => {} // comment, obj_info, etc.
        }
    }

    // ---- locate vertex and face elements ----
    let vp = elems
        .iter()
        .position(|e| e.name == "vertex")
        .context("no vertex element")?;
    let fp = elems
        .iter()
        .position(|e| e.name == "face")
        .context("no face element")?;
    let num_vtx = elems[vp].count;
    let num_face = elems[fp].count;

    // Scalar prop index by name
    let find_scalar = |props: &[PlyProp], name: &str| -> anyhow::Result<usize> {
        props
            .iter()
            .position(|p| {
                if let PlyProp::Scalar { name: n, .. } = p {
                    n.as_str() == name
                } else {
                    false
                }
            })
            .with_context(|| format!("vertex property '{name}' not found"))
    };

    // Byte offset to property i (scalars only, list = 0)
    let scalar_offset = |props: &[PlyProp], up_to: usize| -> usize {
        props[..up_to]
            .iter()
            .map(|p| {
                if let PlyProp::Scalar { ty, .. } = p {
                    ty.size()
                } else {
                    0
                }
            })
            .sum()
    };

    let vtx_props = &elems[vp].props;
    let ix = find_scalar(vtx_props, "x")?;
    let iy = find_scalar(vtx_props, "y")?;
    let iz = find_scalar(vtx_props, "z")?;
    let ox = scalar_offset(vtx_props, ix);
    let oy = scalar_offset(vtx_props, iy);
    let oz = scalar_offset(vtx_props, iz);
    let vtx_rec: usize = vtx_props
        .iter()
        .map(|p| {
            if let PlyProp::Scalar { ty, .. } = p {
                ty.size()
            } else {
                0
            }
        })
        .sum();
    let (x_ty, y_ty, z_ty) = match (&vtx_props[ix], &vtx_props[iy], &vtx_props[iz]) {
        (
            PlyProp::Scalar { ty: a, .. },
            PlyProp::Scalar { ty: b, .. },
            PlyProp::Scalar { ty: c, .. },
        ) => (a.clone(), b.clone(), c.clone()),
        _ => anyhow::bail!("x/y/z must be scalar properties"),
    };

    // Vertex-indices list property in face element
    let face_props = &elems[fp].props;
    let flp = face_props
        .iter()
        .position(|p| {
            if let PlyProp::List { name, .. } = p {
                name == "vertex_indices" || name == "vertex_index"
            } else {
                false
            }
        })
        .context("no vertex_indices list in face element")?;
    let (face_cnt_ty, face_val_ty) = match &face_props[flp] {
        PlyProp::List { cnt_ty, val_ty, .. } => (cnt_ty.clone(), val_ty.clone()),
        _ => unreachable!(),
    };
    // Bytes to skip before the list (scalar props preceding it)
    let face_prefix_bytes: usize = scalar_offset(face_props, flp);
    // Token count to skip before the list count token (ASCII)
    let face_prefix_toks: usize = face_props[..flp]
        .iter()
        .filter(|p| matches!(p, PlyProp::Scalar { .. }))
        .count();
    // Props after the list (for skipping remaining bytes in binary)
    let face_suffix: Vec<PlyProp> = face_props[flp + 1..].to_vec();

    // ---- read data ----
    let mut vtx2xyz: Vec<[Real; 3]> = Vec::with_capacity(num_vtx);
    let mut tri2vtx: Vec<[Index; 3]> = Vec::with_capacity(num_face);

    if is_ascii {
        for (ei, elem) in elems.iter().enumerate() {
            for _ in 0..elem.count {
                line.clear();
                rdr.read_line(&mut line)?;
                let t: Vec<&str> = line.split_whitespace().collect();
                if ei == vp {
                    let x = x_ty.parse_ascii(t[ix])?;
                    let y = y_ty.parse_ascii(t[iy])?;
                    let z = z_ty.parse_ascii(t[iz])?;
                    vtx2xyz.push([x.as_(), y.as_(), z.as_()]);
                } else if ei == fp {
                    let cnt = face_cnt_ty.parse_ascii(t[face_prefix_toks])? as usize;
                    let base = face_prefix_toks + 1;
                    if cnt >= 3 {
                        let i0 = face_val_ty.parse_ascii(t[base])? as usize;
                        for j in 1..cnt - 1 {
                            let ij = face_val_ty.parse_ascii(t[base + j])? as usize;
                            let ij1 = face_val_ty.parse_ascii(t[base + j + 1])? as usize;
                            tri2vtx.push([i0.as_(), ij.as_(), ij1.as_()]);
                        }
                    }
                }
                // other elements: line already consumed
            }
        }
    } else {
        // binary
        let mut rec = vec![0u8; vtx_rec];
        let mut tmp = [0u8; 8];

        for (ei, elem) in elems.iter().enumerate() {
            if ei == vp {
                for _ in 0..elem.count {
                    rdr.read_exact(&mut rec)?;
                    let x = x_ty.read_f64(&rec[ox..], is_le);
                    let y = y_ty.read_f64(&rec[oy..], is_le);
                    let z = z_ty.read_f64(&rec[oz..], is_le);
                    vtx2xyz.push([x.as_(), y.as_(), z.as_()]);
                }
            } else if ei == fp {
                for _ in 0..elem.count {
                    // skip scalar props before the list
                    if face_prefix_bytes > 0 {
                        rdr.read_exact(&mut tmp[..face_prefix_bytes])?;
                    }
                    // list: count then values
                    rdr.read_exact(&mut tmp[..face_cnt_ty.size()])?;
                    let cnt = face_cnt_ty.read_f64(&tmp, is_le) as usize;
                    let vsz = face_val_ty.size();
                    let mut verts = Vec::with_capacity(cnt);
                    for _ in 0..cnt {
                        rdr.read_exact(&mut tmp[..vsz])?;
                        verts.push(face_val_ty.read_f64(&tmp, is_le) as usize);
                    }
                    // skip any suffix props
                    for prop in &face_suffix {
                        match prop {
                            PlyProp::Scalar { ty, .. } => {
                                rdr.read_exact(&mut tmp[..ty.size()])?;
                            }
                            PlyProp::List { cnt_ty, val_ty, .. } => {
                                rdr.read_exact(&mut tmp[..cnt_ty.size()])?;
                                let c = cnt_ty.read_f64(&tmp, is_le) as usize;
                                for _ in 0..c {
                                    rdr.read_exact(&mut tmp[..val_ty.size()])?;
                                }
                            }
                        }
                    }
                    // fan triangulation
                    if cnt >= 3 {
                        let i0 = verts[0];
                        for j in 1..cnt - 1 {
                            tri2vtx.push([i0.as_(), verts[j].as_(), verts[j + 1].as_()]);
                        }
                    }
                }
            } else {
                // skip element we don't need
                for _ in 0..elem.count {
                    for prop in &elem.props {
                        match prop {
                            PlyProp::Scalar { ty, .. } => {
                                rdr.read_exact(&mut tmp[..ty.size()])?;
                            }
                            PlyProp::List { cnt_ty, val_ty, .. } => {
                                rdr.read_exact(&mut tmp[..cnt_ty.size()])?;
                                let c = cnt_ty.read_f64(&tmp, is_le) as usize;
                                for _ in 0..c {
                                    rdr.read_exact(&mut tmp[..val_ty.size()])?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok((tri2vtx, vtx2xyz))
}

#[test]
fn test0() {
    let path_src = std::path::Path::new("../asset/angel.ply");
    let (tri2vtx, vtx2xyz) = load_as_tri_mesh::<_, u32, f32>(path_src).unwrap();
    assert_eq!(vtx2xyz.len(), 128970);
    assert_eq!(tri2vtx.len(), 257936);
    //
    let path_out = std::path::Path::new("../target/out_del_msh_cpu/angel.obj");
    std::fs::create_dir_all(path_out.parent().unwrap()).unwrap();
    crate::io_wavefront_obj::save_tri2vtx_vtx2xyz(path_out, &tri2vtx, &vtx2xyz).unwrap();
}
