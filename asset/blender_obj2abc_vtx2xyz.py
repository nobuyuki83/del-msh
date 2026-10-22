"""Run with Blender, not system Python.

blender -b --python-exit-code 1 --python blender_pntcloud2abc.py -- \
    '/path/pnt_%04d.obj' --start 0 --output '/path/pnt.abc'

Each frame OBJ contains only vertices (point cloud).  Each point is
represented as an icosphere.  All frames must have the same number of
points.  Coordinates are read as written; no OBJ axis conversion is applied.
"""

import argparse
import math
from array import array
from pathlib import Path
import re
import sys


def read_pointcloud_obj(path):
    points = []
    with path.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                points.append(tuple(map(float, fields[1:4])))
                if len(points[-1]) != 3:
                    raise ValueError(f"{path}:{line_number}: invalid vertex")
    if not points:
        raise ValueError(f"{path}: no vertices found")
    return points


def _normalize(v):
    length = math.sqrt(sum(x * x for x in v))
    return tuple(x / length for x in v)


def make_icosphere(radius=1.0, subdivisions=1):
    """Generate a unit icosphere mesh centered at the origin."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = list(map(_normalize, [
        (-1,  t,  0), ( 1,  t,  0), (-1, -t,  0), ( 1, -t,  0),
        ( 0, -1,  t), ( 0,  1,  t), ( 0, -1, -t), ( 0,  1, -t),
        ( t,  0, -1), ( t,  0,  1), (-t,  0, -1), (-t,  0,  1),
    ]))
    faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    midpoint_cache = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoint_cache:
            m = tuple((verts[a][i] + verts[b][i]) / 2 for i in range(3))
            verts.append(_normalize(m))
            midpoint_cache[key] = len(verts) - 1
        return midpoint_cache[key]

    for _ in range(subdivisions):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    verts = [(x * radius, y * radius, z * radius) for x, y, z in verts]
    return verts, faces


def make_pointcloud_mesh(points, sphere_verts, sphere_faces):
    """Instantiate one sphere per point and merge into a single mesh."""
    nv = len(sphere_verts)
    all_verts = []
    all_faces = []
    for px, py, pz in points:
        off = len(all_verts)
        all_verts.extend((px + sx, py + sy, pz + sz) for sx, sy, sz in sphere_verts)
        all_faces.extend(tuple(j + off for j in f) for f in sphere_faces)
    return all_verts, all_faces


def get_sequence(input_path, start):
    """Accept pnt_%04d.obj, or a first file such as pnt_0000.obj."""
    path = Path(input_path).expanduser().resolve()
    name = path.name
    placeholders = list(re.finditer(r"%0?\d*d", name))
    if placeholders:
        if len(placeholders) != 1:
            raise ValueError("Use exactly one %d or %04d placeholder")
        match = placeholders[0]
        token = match.group()
        number = 0 if start is None else start

        def make_path(i):
            return path.parent / (
                name[: match.start()] + (token % i) + name[match.end():]
            )
    else:
        match = re.search(r"(\d+)(?=\.obj$)", name, re.IGNORECASE)
        if match is None:
            raise ValueError(
                "Pass pnt_%04d.obj or a numbered file such as pnt_0000.obj"
            )
        width = len(match.group())
        number = int(match.group()) if start is None else start

        def make_path(i):
            return path.parent / (
                name[: match.start()] + f"{i:0{width}d}" + name[match.end():]
            )

    paths = []
    while make_path(number).is_file():
        paths.append(make_path(number))
        number += 1
    if not paths:
        raise FileNotFoundError(make_path(number))
    print(f"Found {len(paths)} frames; stopped at missing file: {make_path(number)}")
    return paths


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj_path", help="First numbered OBJ, or pattern pnt_%04d.obj")
    parser.add_argument(
        "--start", type=int, default=None, help="First file index; pattern default: 0"
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--radius", type=float, default=0.02, help="Sphere radius per point")
    parser.add_argument("--subdiv", type=int, default=1, help="Icosphere subdivisions (1=80 tris, 2=320 tris)")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.radius <= 0:
        parser.error("--radius must be positive")

    paths = get_sequence(args.obj_path, args.start)
    output = (args.output or paths[0].with_suffix(".abc")).expanduser().resolve()
    if output.suffix.lower() != ".abc":
        parser.error("Output must have .abc extension")
    if not bpy.app.build_options.alembic:
        raise RuntimeError("This Blender build does not support Alembic")

    sphere_verts, sphere_faces = make_icosphere(args.radius, args.subdiv)
    print(f"Sphere: {len(sphere_verts)} verts, {len(sphere_faces)} faces, radius={args.radius}, subdiv={args.subdiv}")

    points0 = read_pointcloud_obj(paths[0])
    vertices, faces = make_pointcloud_mesh(points0, sphere_verts, sphere_faces)
    print(f"Frame 1: {len(points0)} points → {len(vertices)} verts, {len(faces)} faces")

    scene = bpy.data.scenes.new("PntCloudExport")
    bpy.context.window.scene = scene
    scene.render.fps = args.fps
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = len(paths)

    mesh = bpy.data.meshes.new("PointCloudMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("PointCloud", mesh)
    scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    basis = obj.shape_key_add(name="Basis", from_mix=False)
    keys = mesh.shape_keys
    keys.use_relative = False
    basis.interpolation = "KEY_LINEAR"
    keys.eval_time = basis.frame
    keys.keyframe_insert(data_path="eval_time", frame=1)

    for frame, path in enumerate(paths[1:], 2):
        points = read_pointcloud_obj(path)
        if len(points) != len(points0):
            raise ValueError(f"{path}: point count changed ({len(points)} != {len(points0)})")
        frame_verts, _ = make_pointcloud_mesh(points, sphere_verts, sphere_faces)
        key = obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
        key.data.foreach_set("co", array("f", (v for p in frame_verts for v in p)))
        key.interpolation = "KEY_LINEAR"
        keys.eval_time = key.frame
        keys.keyframe_insert(data_path="eval_time", frame=frame)
        print(f"Loaded {frame}/{len(paths)}: {path.name}")

    scene.frame_set(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.wm.alembic_export(
        filepath=str(output),
        check_existing=False,
        start=1,
        end=len(paths),
        selected=True,
        xsamples=1,
        gsamples=1,
        sh_open=0.0,
        sh_close=0.0,
        uvs=False,
        normals=True,
        export_hair=False,
        export_particles=False,
        as_background_job=False,
    )
    if "FINISHED" not in result or not output.is_file():
        raise RuntimeError(f"Alembic export failed: {result}")
    print(f"Saved: {output} ({len(paths)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
