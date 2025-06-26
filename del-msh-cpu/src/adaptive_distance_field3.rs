pub trait SignedDistanceField3<Real>
{
    fn sdf(&self, x: Real, y: Real, z: Real) -> Real;
}

#[derive(Clone, Debug)]
pub struct Node<Real>
    where Real: num_traits::Float
{
    cent: [Real; 3],         // セル中心座標
    hw: Real,                // half-width
    corner_dist: [Real; 8],  // 8 コーナーの SDF 値
    child_idxs: [usize; 8], // 子ノードのインデックス (-1: なし)
}

impl<Real> Node<Real>
where Real:num_traits::Float
{
    /// 8 頂点の SDF を一括で計算
    pub fn set_corner_dist<S: SignedDistanceField3<Real>>(&mut self, ct: &S) {
        let [cx, cy, cz] = self.cent;
        let hw = self.hw;
        self.corner_dist = [
            ct.sdf(cx - hw, cy - hw, cz - hw),
            ct.sdf(cx + hw, cy - hw, cz - hw),
            ct.sdf(cx + hw, cy + hw, cz - hw),
            ct.sdf(cx - hw, cy + hw, cz - hw),
            ct.sdf(cx - hw, cy - hw, cz + hw),
            ct.sdf(cx + hw, cy - hw, cz + hw),
            ct.sdf(cx + hw, cy + hw, cz + hw),
            ct.sdf(cx - hw, cy + hw, cz + hw),
        ];
    }
}

/// 必要に応じて 8 分木を細分化
pub fn make_child_tree<Real, S: SignedDistanceField3<Real>>(
    ct: &S,
    nodes: &mut Vec<Node<Real>>, // 生成した子ノードを push するバッファ
    i_node: usize,
    min_hw: Real,
    max_hw: Real,
)
where Real: num_traits::Float + 'static,
      f64: num_traits::AsPrimitive<Real>
{
    let zero = Real::zero();
    let one = Real::one();
    let half = Real::one() / (Real::one() + Real::one());
    let one4th = half * half;
    let one5th = one / (one + one + one + one + one);
    let one8th = one4th * one4th;
    let one_point_eight = one + one - one5th;
    let zero_point_eight = one - one5th;
    // ───────── step-0: 早期終了１ ─────────
    if nodes[i_node].hw * half < min_hw {
        nodes[i_node].child_idxs[0] = usize::MAX;
        return;
    }

    let [cx, cy, cz] = nodes[i_node].cent;
    let hw = nodes[i_node].hw;

    // ───────── step-1: 辺・面・中心の SDF を計算 ─────────
    macro_rules! sdf {
        ($dx:expr, $dy:expr, $dz:expr) => {
            ct.sdf(cx + $dx, cy + $dy, cz + $dz)
        };
    }

    // Edges
    let va100 = sdf!(zero, -hw, -hw);
    let va210 = sdf!(hw, zero, -hw);
    let va120 = sdf!(zero, hw, -hw);
    let va010 = sdf!(-hw, zero, -hw);

    let va001 = sdf!(-hw, -hw, zero);
    let va201 = sdf!(hw, -hw, zero);
    let va221 = sdf!(hw, hw, zero);
    let va021 = sdf!(-hw, hw, zero);

    let va102 = sdf!(zero, -hw, hw);
    let va212 = sdf!(hw, zero, hw);
    let va122 = sdf!(zero, hw, hw);
    let va012 = sdf!(-hw, zero, hw);

    // Faces
    let va101 = sdf!(zero, -hw, zero);
    let va211 = sdf!(hw, zero, zero);
    let va121 = sdf!(zero, hw, zero);
    let va011 = sdf!(-hw, zero, zero);
    let va110 = sdf!(zero, zero, -hw);
    let va112 = sdf!(zero, zero, hw);

    // Center
    let va111 = sdf!(zero, zero, zero);

    // ───────── step-2: 細かくし過ぎを抑制 ─────────
    let mut need_child = false;

    if nodes[i_node].hw * half > max_hw {
        need_child = true; // 上限を超えているので細分化を許可
    } else {
        // 最小絶対距離が閾値より大きいならメッシュは無い
        let min_dist = [
            va111, va100, va210, va120, va010, va001, va201, va221, va021, va101, va211, va121,
            va011, va102, va212, va122, va012, va110, va112,
        ]
        .iter()
        .map(|v| v.abs())
        .fold(Real::infinity(), Real::min);

        if min_dist > hw * one_point_eight {
            nodes[i_node].child_idxs[0] = usize::MAX;
            return;
        }

        if min_dist < min_hw {
            need_child = true;
        } else {
            // ───────── step-3: 補間誤差チェック ─────────
            let interp = |v0, v1| half * (v0 + v1);
            let interp4 = |v0, v1, v2, v3| one4th * (v0 + v1 + v2 + v3);

            macro_rules! check {
                ($cond:expr) => {{
                    if ($cond).abs() > min_hw * zero_point_eight {
                        need_child = true;
                    }
                }};
            }

            let p_dist = &nodes[i_node].corner_dist;

            // 12 辺
            check!(va100 - interp(p_dist[0], p_dist[1]));
            check!(va210 - interp(p_dist[1], p_dist[2]));
            check!(va120 - interp(p_dist[2], p_dist[3]));
            check!(va010 - interp(p_dist[3], p_dist[0]));
            check!(va102 - interp(p_dist[4], p_dist[5]));
            check!(va212 - interp(p_dist[5], p_dist[6]));
            check!(va122 - interp(p_dist[6], p_dist[7]));
            check!(va012 - interp(p_dist[7], p_dist[4]));
            check!(va001 - interp(p_dist[0], p_dist[4]));
            check!(va201 - interp(p_dist[1], p_dist[5]));
            check!(va221 - interp(p_dist[2], p_dist[6]));
            check!(va021 - interp(p_dist[3], p_dist[7]));

            // 6 面中央
            check!(va101 - interp4(p_dist[0], p_dist[1], p_dist[4], p_dist[5]));
            check!(va211 - interp4(p_dist[1], p_dist[2], p_dist[5], p_dist[6]));
            check!(va121 - interp4(p_dist[2], p_dist[3], p_dist[6], p_dist[7]));
            check!(va011 - interp4(p_dist[3], p_dist[0], p_dist[7], p_dist[4]));
            check!(va110 - interp4(p_dist[0], p_dist[1], p_dist[2], p_dist[3]));
            check!(va112 - interp4(p_dist[4], p_dist[5], p_dist[6], p_dist[7]));

            // 立方体中心
            let center_interp = one8th * (p_dist[0] + p_dist[1] + p_dist[2] + p_dist[3] + p_dist[4] + p_dist[5] + p_dist[6] + p_dist[7]);
            check!(va111 - center_interp);
        }
    }
    // ───────── step-4: 子ノード生成 ─────────
    if !need_child {
        nodes[i_node].child_idxs[0] = usize::MAX; // 分割不要
        return;
    }
    // eight children
    nodes[i_node].child_idxs = [usize::MAX; 8]; // まず -1 で初期化
    use del_geo_core::vec3::Vec3;
    for i_node_hex in 0..8 {
        let offset = del_geo_core::vec3::cast::<f64, Real>( &del_geo_core::hex::HEX_SIGN[i_node_hex] );
        let p_dist = &nodes[i_node].corner_dist; // distances of parent node
        let c_dist = match i_node_hex {
            0 => [p_dist[0], va100, va110, va010, va001, va101, va111, va011],
            1 => [va100, p_dist[1], va210, va110, va101, va201, va211, va111],
            2 => [va110, va210, p_dist[2], va120, va111, va211, va221, va121],
            3 => [va010, va110, va120, p_dist[3], va011, va111, va121, va021],
            4 => [va001, va101, va111, va011, p_dist[4], va102, va112, va012],
            5 => [va101, va201, va211, va111, va102, p_dist[5], va212, va112],
            6 => [va111, va211, va221, va121, va112, va212, p_dist[6], va122],
            7 => [va011, va111, va121, va021, va012, va112, va122, p_dist[7]],
            _ => unreachable!(),
        };
        let child = Node {
            cent: offset.scale(hw * half).add(&[cx, cy, cz]),
            hw: hw * half,
            corner_dist: c_dist,
            child_idxs: [usize::MAX; 8],
        };
        // ベクタに追加してインデックスを保持
        let idx = nodes.len();
        nodes.push(child);
        nodes[i_node].child_idxs[i_node_hex] = idx;
    }
    for j_node in nodes[i_node].child_idxs {
        make_child_tree(ct, nodes, j_node, min_hw, max_hw);
    }
}

#[test]
fn hoge() {
    struct Sphere {}
    impl SignedDistanceField3<f64> for Sphere
    {
        fn sdf(&self, x: f64, y: f64, z: f64) -> f64 {
            0.33 - (x * x + y * y + z * z).sqrt()
        }
    }

    let sphere = Sphere {};
    let hw = 1.0;
    let cent = [0., 0., 0.];
    let corner_dist = {
        let mut corner_dist = [0f64; 8];
        for i_node in 0..8 {
            let d = del_geo_core::hex::HEX_SIGN[i_node];
            use del_geo_core::vec3::Vec3;
            let pos = d.scale(hw).add(&cent);
            let dist = sphere.sdf(pos[0], pos[1], pos[2]);
            corner_dist[i_node] = dist;
        }
        corner_dist
    };
    let node0 = Node {
        hw,
        cent,
        child_idxs: [usize::MAX; 8],
        corner_dist,
    };
    let mut nodes = vec![node0];
    make_child_tree(&sphere, &mut nodes, 0, 0.05, 0.26);
    dbg!(nodes.len());

    let mut tri2xyz = vec![];
    for node in nodes {
        if node.child_idxs[0] != usize::MAX {
            continue;
        }
        del_geo_core::hex::iso_surface(&mut tri2xyz, node.cent, node.hw, &node.corner_dist);
    }
    {
        use slice_of_array::SliceFlatExt;
        crate::io_obj::save_tri2xyz("../target/sdf.obj", tri2xyz.flat()).unwrap();
    }
}
