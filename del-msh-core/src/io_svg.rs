//! methods for SVG file format

pub fn svg_outline_path_from_shape(s0: &str) -> Vec<String> {
    let s0 = s0.as_bytes();
    let mut imark = 0;
    let mut strs = Vec::<String>::new();
    for i in 0..s0.len() {
        if s0[i].is_ascii_digit() {
            continue;
        }
        if s0[i] == b',' {
            strs.push(std::str::from_utf8(&s0[imark..i]).unwrap().to_string());
            imark = i + 1; // mark should be the beginning position of the string so move next
        }
        if s0[i] == b' ' {
            // sometimes the space act as delimiter in the SVG (inkscape version)
            if i > imark {
                strs.push(std::str::from_utf8(&s0[imark..i]).unwrap().to_string());
            }
            imark = i + 1; // mark should be the beginning position of the string so move next
        }
        if s0[i] == b'-' {
            if i > imark {
                strs.push(std::str::from_utf8(&s0[imark..i]).unwrap().to_string());
            }
            imark = i;
        }
        if s0[i].is_ascii_alphabetic() {
            if i > imark {
                strs.push(std::str::from_utf8(&s0[imark..i]).unwrap().to_string());
            }
            strs.push(std::str::from_utf8(&[s0[i]]).unwrap().to_string()); // push tag
            imark = i + 1;
        }
    }
    if s0.len() > imark {
        strs.push(
            std::str::from_utf8(&s0[imark..s0.len()])
                .unwrap()
                .to_string(),
        );
    }
    strs
}

#[allow(clippy::identity_op)]
pub fn svg_loops_from_outline_path(strs: &[String]) -> Vec<(Vec<[f32; 2]>, Vec<usize>, bool)> {
    use del_geo_core::vec2::Vec2;
    let hoge = |s0: &str, s1: &str| [s0.parse::<f32>().unwrap(), s1.parse::<f32>().unwrap()];
    let mut loops: Vec<(Vec<[f32; 2]>, Vec<usize>, bool)> = vec![];
    let mut vtxl2xy: Vec<[f32; 2]> = vec![];
    let mut seg2vtxl: Vec<usize> = vec![0];
    assert!(strs[0] == "M" || strs[0] == "m");
    // assert!(strs[strs.len() - 1] == "Z" || strs[strs.len() - 1] == "z");
    let mut pos_cur = [0f32; 2];
    let mut is = 0;
    loop {
        if strs[is] == "M" {
            // start absolute
            is += 1;
            pos_cur = hoge(&strs[is + 0], &strs[is + 1]);
            vtxl2xy.push(pos_cur);
            is += 2;
        } else if strs[is] == "m" {
            // start relative
            is += 1;
            pos_cur = pos_cur.add(&hoge(&strs[is + 0], &strs[is + 1]));
            vtxl2xy.push(pos_cur);
            is += 2;
        } else if strs[is] == "l" {
            // line relative
            is += 1;
            loop {
                seg2vtxl.push(vtxl2xy.len());
                let p1 = pos_cur.add(&hoge(&strs[is + 0], &strs[is + 1]));
                vtxl2xy.push(p1);
                pos_cur = p1;
                is += 2;
                if strs[is].as_bytes()[0].is_ascii_alphabetic() {
                    break;
                }
            }
        } else if strs[is] == "L" {
            // line absolute
            is += 1;
            loop {
                seg2vtxl.push(vtxl2xy.len());
                let p1 = hoge(&strs[is + 0], &strs[is + 1]);
                vtxl2xy.push(p1);
                pos_cur = p1;
                is += 2;
                if strs[is].as_bytes()[0].is_ascii_alphabetic() {
                    break;
                }
            }
        } else if strs[is] == "v" {
            // vertical relative
            seg2vtxl.push(vtxl2xy.len());
            let p1 = pos_cur.add(&[0., strs[is + 1].parse::<f32>().unwrap()]);
            vtxl2xy.push(p1);
            pos_cur = p1;
            is += 2;
        } else if strs[is] == "V" {
            // vertical absolute
            seg2vtxl.push(vtxl2xy.len());
            let p1 = [pos_cur[0], strs[is + 1].parse::<f32>().unwrap()];
            vtxl2xy.push(p1);
            pos_cur = p1;
            is += 2;
        } else if strs[is] == "H" {
            // horizontal absolute
            seg2vtxl.push(vtxl2xy.len());
            let p1 = [strs[is + 1].parse::<f32>().unwrap(), pos_cur[1]];
            vtxl2xy.push(p1);
            pos_cur = p1;
            is += 2;
        } else if strs[is] == "h" {
            // horizontal relative
            seg2vtxl.push(vtxl2xy.len());
            let dh = strs[is + 1].parse::<f32>().unwrap();
            let p1 = pos_cur.add(&[dh, 0.]);
            vtxl2xy.push(p1);
            pos_cur = p1;
            is += 2;
        } else if strs[is] == "q" {
            // relative
            is += 1;
            loop {
                // loop for poly quadratic Bézeir curve
                let pm0 = pos_cur.add(&hoge(&strs[is + 0], &strs[is + 1]));
                let p1 = pos_cur.add(&hoge(&strs[is + 2], &strs[is + 3]));
                vtxl2xy.push(pm0);
                seg2vtxl.push(vtxl2xy.len());
                vtxl2xy.push(p1);
                pos_cur = p1;
                is += 4;
                if is == strs.len() {
                    loops.push((vtxl2xy.clone(), seg2vtxl.clone(), false));
                    break;
                }
                if strs[is].as_bytes()[0].is_ascii_alphabetic() {
                    break;
                }
            }
        } else if strs[is] == "Q" {
            // absolute
            is += 1;
            loop {
                // loop for poly-Bezeir curve
                let pm0 = hoge(&strs[is + 0], &strs[is + 1]);
                let p1 = hoge(&strs[is + 2], &strs[is + 3]);
                vtxl2xy.push(pm0);
                seg2vtxl.push(vtxl2xy.len());
                vtxl2xy.push(p1);
                pos_cur = p1;
                is += 4;
                if strs[is].as_bytes()[0].is_ascii_alphabetic() {
                    break;
                }
            }
        } else if strs[is] == "c" {
            // relative
            is += 1;
            loop {
                // loop for poly cubic Bézeir curve
                let pm0 = pos_cur.add(&hoge(&strs[is + 0], &strs[is + 1]));
                let pm1 = pos_cur.add(&hoge(&strs[is + 2], &strs[is + 3]));
                let p1 = pos_cur.add(&hoge(&strs[is + 4], &strs[is + 5]));
                vtxl2xy.push(pm0);
                vtxl2xy.push(pm1);
                seg2vtxl.push(vtxl2xy.len());
                vtxl2xy.push(p1);
                pos_cur = p1;
                is += 6;
                if is == strs.len() {
                    loops.push((vtxl2xy.clone(), seg2vtxl.clone(), false));
                    break;
                }
                if strs[is].as_bytes()[0].is_ascii_alphabetic() {
                    break;
                }
            }
        } else if strs[is] == "z" || strs[is] == "Z" {
            let pe = vtxl2xy[0];
            let ps = vtxl2xy[vtxl2xy.len() - 1];
            let _dist0 = ps.sub(&pe).norm();
            loops.push((vtxl2xy.clone(), seg2vtxl.clone(), true));
            vtxl2xy.clear();
            seg2vtxl = vec![0];
            is += 1;
        } else {
            dbg!("error!--> ", &strs[is]);
            break;
        }
        if is == strs.len() {
            break;
        }
    }
    loops
}

pub fn polybezier2polyloop(
    vtx2xy: &[[f32; 2]],
    seg2vtx: &[usize],
    is_close: bool,
    edge_length: f32,
) -> Vec<[f32; 2]> {
    use del_geo_core::vec2::Vec2;
    let mut ret: Vec<[f32; 2]> = vec![];
    let num_seg = seg2vtx.len() - 1;
    for i_seg in 0..num_seg {
        let (is_vtx, ie_vtx) = (seg2vtx[i_seg], seg2vtx[i_seg + 1]);
        let ps = &vtx2xy[is_vtx];
        let pe = &vtx2xy[ie_vtx];
        if ie_vtx - is_vtx == 1 {
            let len = pe.sub(ps).norm();
            let ndiv = (len / edge_length) as usize;
            for i in 0..ndiv {
                let r = i as f32 / ndiv as f32;
                let p = ps.scale(1f32 - r).add(&pe.scale(r));
                ret.push(p);
            }
        } else if ie_vtx - is_vtx == 2 {
            // quadratic bezier
            let pc = &vtx2xy[is_vtx + 1];
            let ndiv = 10;
            for idiv in 0..ndiv {
                let t0 = idiv as f32 / ndiv as f32;
                let p0 = del_geo_core::bezier_quadratic::eval(ps, pc, pe, t0);
                ret.push(p0);
            }
        } else if ie_vtx - is_vtx == 3 {
            // cubic bezier
            let pc0 = &vtx2xy[is_vtx + 1];
            let pc1 = &vtx2xy[is_vtx + 2];
            let samples = del_geo_core::bezier_cubic::sample_uniform_length(
                del_geo_core::bezier_cubic::ControlPoints::<'_, f32, 2> {
                    p0: ps,
                    p1: pc0,
                    p2: pc1,
                    p3: pe,
                },
                edge_length,
                true,
                false,
                30,
            );
            ret.extend(samples);
        }
    }
    if is_close {
        let ps = &vtx2xy[vtx2xy.len() - 1];
        let pe = &vtx2xy[0];
        let len = pe.sub(ps).norm();
        let ndiv = (len / edge_length) as usize;
        for i in 0..ndiv {
            let r = i as f32 / ndiv as f32;
            let p = ps.scale(1f32 - r).add(&pe.scale(r));
            ret.push(p);
        }
    }
    ret
}

#[test]
fn hoge() {
    let str2 = "M 457.60409,474.77081 H 347.66161 L 208.25942,282.21963 \
    q -15.48914,0.60741 -25.20781,0.60741 -3.94821,0 -8.50384,0 -4.55562,-0.3037 -9.41496,-0.60741 \
    v 119.66114 q 0,38.87469 8.50384,48.28965 11.54092,13.36318 34.62277,13.36318 h 16.09655 \
    v 11.23721 H 47.901331 V 463.5336 h 15.489133 \
    q 26.118931,0 37.356146,-17.00768 6.37788,-9.41496 6.37788,-44.64515 V 135.83213 \
    q 0,-38.874683 -8.50384,-48.289646 Q 86.776018,74.17931 63.390464,74.17931 H 47.901331 \
    V 62.942096 H 197.93333 q 65.60103,0 96.5793,9.718671 31.28197,9.414964 52.84528,35.230183 \
    21.86701,25.51152 21.86701,61.04541 0,37.96356 -24.9041,65.90474 -24.60039,27.94118 \
    -76.53454,39.48211 l 85.03838,118.1426 q 29.15601,40.69694 50.1119,54.06011 20.95589,13.36318 \
    54.66753,17.00768 z \
    M 165.13281,263.08599 q 5.77046,0 10.02238,0.30371 4.25192,0 6.9853,0 \
    58.91944,0 88.68288,-25.51151 30.06714,-25.51152 30.06714,-64.99362 0,-38.57098 \
    -24.29668,-62.56395 -23.99297,-24.296679 -63.77879,-24.296679 -17.61509,0 -47.68223,5.770461 z";
    let strs = svg_outline_path_from_shape(str2);
    let loops = svg_loops_from_outline_path(&strs);
    // dbg!(&loops[0].0);
    for polybezier in loops {
        polybezier2polyloop(&polybezier.0, &polybezier.1, polybezier.2, 5.0);
    }
}

#[test]
fn hoge2() {
    let str0 = "M2792 12789 c-617 -83 -1115 -568 -1244 -1212 -32 -160 -32 -443 0 \
    -602 76 -382 282 -720 573 -938 l36 -27 -58 -172 c-90 -269 -174 -590 -216 \
    -833 -13 -76 -17 -159 -17 -345 -1 -212 2 -261 21 -360 43 -225 113 -418 217 \
    -601 204 -356 574 -691 972 -880 96 -45 103 -51 160 -126 32 -43 105 -126 162 \
    -185 l103 -106 -47 -44 c-143 -131 -352 -391 -469 -584 -306 -501 -465 -1076 \
    -501 -1807 -5 -117 -9 -137 -23 -137 -38 0 -104 26 -211 85 -440 240 -827 302 \
    -1345 215 -216 -37 -301 -67 -409 -144 -258 -186 -410 -530 -476 -1081 -17 \
    -143 -24 -516 -11 -655 42 -486 188 -848 446 -1105 208 -209 459 -325 790 \
    -366 110 -13 513 -17 615 -6 l65 8 24 -63 c132 -354 523 -580 1149 -668 252 \
    -35 395 -44 722 -44 258 -1 351 3 450 17 134 19 295 54 400 89 74 23 256 107 \
    297 136 27 18 34 18 133 5 150 -19 624 -28 731 -14 84 11 86 11 150 -18 298 \
    -135 701 -204 1259 -218 280 -6 462 4 662 38 459 78 788 280 941 577 25 48 51 \
    106 58 130 9 31 16 41 28 37 41 -12 362 -26 491 -20 388 17 612 78 837 228 \
    336 223 534 574 615 1092 19 119 22 181 22 420 0 285 -12 437 -51 635 -14 73 \
    -20 89 -48 112 -24 20 -40 51 -65 120 -79 227 -184 405 -319 539 -130 130 \
    -226 178 -463 233 -188 44 -247 51 -438 50 -152 0 -203 -4 -286 -22 -199 -43 \
    -339 -101 -579 -239 -77 -44 -158 -86 -180 -92 -44 -12 -170 -14 -251 -5 l-51 \
    6 -6 257 c-8 352 -37 606 -102 896 -95 423 -268 810 -513 1146 l-41 56 39 38 \
    c37 36 40 37 127 42 478 24 909 263 1196 664 103 143 213 372 285 590 148 450 \
    215 839 216 1264 l1 230 65 32 c246 121 482 349 628 608 267 473 263 1087 -10 \
    1559 -215 371 -560 622 -978 712 -117 25 -398 26 -522 1 -200 -40 -417 -137 \
    -576 -257 -52 -38 -95 -70 -97 -70 -2 0 -45 30 -95 66 -389 280 -904 530 \
    -1298 629 -116 29 -289 57 -507 82 -229 26 -799 26 -1000 0 -265 -35 -499 -87 \
    -714 -159 l-124 -42 -35 44 c-75 95 -259 267 -350 328 -157 105 -323 175 -500 \
    212 -114 24 -350 34 -460 19z";
    let strs = svg_outline_path_from_shape(str0);
    let loops = svg_loops_from_outline_path(&strs);
    assert_eq!(loops.len(), 1);
    let polyline = polybezier2polyloop(&loops[0].0, &loops[0].1, loops[0].2, 10.0);
    use slice_of_array::SliceFlatExt;
    let polyline = polyline.flat().to_vec();
    let polyline = crate::polyloop::resample::<_, 2>(&polyline, 100);
    crate::io_obj::save_vtx2xyz_as_polyloop("../target/svg.obj", &polyline, 2).unwrap();
}

#[test]
fn hoge3() {
    let str_path = "M7920 11494 c-193 -21 -251 -29 -355 -50 -540 -105 -1036 -366 -1442 \
    -758 -515 -495 -834 -1162 -904 -1891 -15 -154 -6 -563 15 -705 66 -440 220 \
    -857 442 -1203 24 -37 44 -69 44 -71 0 -2 -147 -3 -327 -4 -414 -1 -765 -23 \
    -1172 -72 -97 -12 -167 -17 -170 -11 -3 5 -33 52 -66 106 -231 372 -633 798 \
    -1040 1101 -309 229 -625 409 -936 532 -287 113 -392 130 -500 79 -65 -32 \
    -118 -81 -249 -237 -627 -745 -1009 -1563 -1170 -2505 -54 -320 -77 -574 -86 \
    -965 -28 -1207 238 -2308 785 -3242 120 -204 228 -364 270 -397 84 -67 585 \
    -319 901 -454 1197 -511 2535 -769 3865 -744 983 19 1875 166 2783 458 334 \
    108 918 340 1013 404 99 65 407 488 599 824 620 1080 835 2329 614 3561 -75 \
    415 -226 892 -401 1262 -39 82 -54 124 -47 133 5 7 42 58 82 114 41 55 77 99 \
    81 96 4 -2 68 -8 142 -14 766 -53 1474 347 1858 1051 105 192 186 439 228 693 \
    27 167 24 487 -6 660 -33 189 -64 249 -150 289 -46 21 -51 21 -846 21 -440 0 \
    -828 -3 -861 -7 l-62 -7 -32 86 c-54 143 -194 412 -289 554 -479 720 -1201 \
    1178 -2040 1295 -101 14 -496 27 -571 18z";
    let outline_path = svg_outline_path_from_shape(str_path);
    // dbg!(&outline_path);
    let loops = svg_loops_from_outline_path(&outline_path);
    let vtxl2xy = polybezier2polyloop(&loops[0].0, &loops[0].1, loops[0].2, 600.);
    use slice_of_array::SliceFlatExt;
    let vtxl2xy = crate::vtx2xy::normalize(vtxl2xy.flat(), &[0.5, 0.5], 1.0);
    crate::io_obj::save_vtx2xyz_as_polyloop("../target/duck_curve.obj", &vtxl2xy, 2).unwrap();
}
