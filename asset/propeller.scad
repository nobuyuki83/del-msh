$fn = 96;

// 寸法（mm）
outer_d = 100;
hub_d = 24;
hub_height = 20;
shaft_hole_d = 5;

blade_count = 3;
blade_thickness = 2;

// 羽根の形
root_chord = 15; //20;
tip_chord = 8; // 12;
middle_bulge = 15; // 20;   // 中ほどを広げる量
root_pitch = 38;
tip_pitch = 18;
tip_sweep = 30;      // 先端を回転方向に曲げる角度
camber = 1.5;        // 羽根の断面の反り

radial_steps = 16;
chord_steps = 8;

module blade() {
    r0 = hub_d/2 - 3;  // ハブに食い込ませる
    r1 = outer_d/2;
    row_size = chord_steps + 1;
    layer_size = (radial_steps + 1) * row_size;

    function idx(i, j, layer) =
        layer * layer_size + i * row_size + j;

    function vertex(i, j, side) =
        let(
            u = i / radial_steps,
            v = j / chord_steps,
            r = r0 + (r1 - r0) * u,
            chord = root_chord * (1-u)
                  + tip_chord * u
                  + middle_bulge * sin(180*u),
            pitch = root_pitch * (1-u) + tip_pitch * u,
            angle = tip_sweep * u*u,
            t = (v - 0.5) * chord
        )
        [
            r*cos(angle) - t*cos(pitch)*sin(angle),
            r*sin(angle) + t*cos(pitch)*cos(angle),
            t*sin(pitch)
                + camber*sin(180*u)*sin(180*v)
                + side*blade_thickness/2
        ];

    points = concat(
        [for (i = [0:radial_steps], j = [0:chord_steps])
            vertex(i, j,  1)],
        [for (i = [0:radial_steps], j = [0:chord_steps])
            vertex(i, j, -1)]
    );

    faces = concat(
        // 上面
        [for (i = [0:radial_steps-1], j = [0:chord_steps-1])
            each [
                [idx(i,j,0), idx(i+1,j,0), idx(i+1,j+1,0)],
                [idx(i,j,0), idx(i+1,j+1,0), idx(i,j+1,0)]
            ]],

        // 下面
        [for (i = [0:radial_steps-1], j = [0:chord_steps-1])
            each [
                [idx(i,j,1), idx(i,j+1,1), idx(i+1,j+1,1)],
                [idx(i,j,1), idx(i+1,j+1,1), idx(i+1,j,1)]
            ]],

        // 羽根の両側の縁
        [for (i = [0:radial_steps-1])
            each [
                [idx(i,0,0), idx(i,0,1), idx(i+1,0,1)],
                [idx(i,0,0), idx(i+1,0,1), idx(i+1,0,0)],
                [idx(i,chord_steps,0), idx(i+1,chord_steps,0),
                                      idx(i+1,chord_steps,1)],
                [idx(i,chord_steps,0), idx(i+1,chord_steps,1),
                                      idx(i,chord_steps,1)]
            ]],

        // 根元と先端
        [for (j = [0:chord_steps-1])
            each [
                [idx(0,j,0), idx(0,j+1,0), idx(0,j+1,1)],
                [idx(0,j,0), idx(0,j+1,1), idx(0,j,1)],
                [idx(radial_steps,j+1,0), idx(radial_steps,j,0),
                                          idx(radial_steps,j,1)],
                [idx(radial_steps,j+1,0), idx(radial_steps,j,1),
                                          idx(radial_steps,j+1,1)]
            ]]
    );

    polyhedron(points = points, faces = faces, convexity = 10);
}

difference() {
    union() {
        cylinder(d = hub_d, h = hub_height, center = true);

        for (i = [0:blade_count-1])
            rotate([0, 0, i*360/blade_count])
                blade();
    }

    cylinder(d = shaft_hole_d, h = hub_height + 2, center = true);
}