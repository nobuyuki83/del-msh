pub fn to_polyhedron_mesh<IDX>(
    tet2vtx: &[[IDX; 4]],
    pyrmd2vtx: &[[IDX; 5]],
    prism2vtx: &[[IDX; 6]],
    hex2vtx: &[[IDX; 8]],
    elem2idx_offset: &mut [IDX],
    idx2vtx: &mut [IDX],
) where
    IDX: num_traits::PrimInt,
{
    let one = IDX::one();
    let four = one + one + one + one;
    let five = four + one;
    let six = five + one;
    let eight = four + four;
    let num_tet = tet2vtx.len();
    let num_pyrmd = pyrmd2vtx.len();
    let num_prism = prism2vtx.len();
    let num_hex = hex2vtx.len();
    let num_elem = num_tet + num_pyrmd + num_prism + num_hex;
    assert_eq!(elem2idx_offset.len(), num_elem + 1);
    assert_eq!(
        idx2vtx.len(),
        tet2vtx.len() * 4 + pyrmd2vtx.len() * 5 + prism2vtx.len() * 6 + hex2vtx.len() * 8
    );
    {
        let mut idx = 0;
        elem2idx_offset[idx] = IDX::zero();
        for _i_tet in 0..num_tet {
            elem2idx_offset[idx + 1] = elem2idx_offset[idx] + four;
            idx += 1;
        }
        for _i_pyrmd in 0..num_pyrmd {
            elem2idx_offset[idx + 1] = elem2idx_offset[idx] + five;
            idx += 1;
        }
        for _i_prism in 0..num_prism {
            elem2idx_offset[idx + 1] = elem2idx_offset[idx] + six;
            idx += 1;
        }
        for _i_hex in 0..num_hex {
            elem2idx_offset[idx + 1] = elem2idx_offset[idx] + eight;
            idx += 1;
        }
        assert_eq!(idx, num_elem);
        elem2idx_offset
    };
    idx2vtx[0..num_tet * 4].copy_from_slice(tet2vtx.as_flattened());
    idx2vtx[num_tet * 4..(num_tet * 4 + num_pyrmd * 5)].copy_from_slice(pyrmd2vtx.as_flattened());
    idx2vtx[(num_tet * 4 + num_pyrmd * 5)..(num_tet * 4 + num_pyrmd * 5 + num_prism * 6)]
        .copy_from_slice(prism2vtx.as_flattened());
    idx2vtx[(num_tet * 4 + num_pyrmd * 5 + num_prism * 6)
        ..(num_tet * 4 + num_pyrmd * 5 + num_prism * 6 + num_hex * 8)]
        .copy_from_slice(hex2vtx.as_flattened());
}
