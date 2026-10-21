"""Run with Blender, not system Python.

blender -b --python-exit-code 1 --python obj_sequence_to_abc.py -- \
    '/path/mesh_%04d.obj' --start 0 --output '/path/mesh.abc'

All frames must have identical vertex order and triangle connectivity.
Geometry only: UVs, materials and OBJ normals are not copied.
Coordinates are read as written; no OBJ axis conversion is applied.
Shape keys store all frames in memory. Intended for moderate sequences.
"""

import argparse
from array import array
from pathlib import Path
import re
import sys


def read_obj(path):
    vertices = []
    faces = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append(tuple(map(float, fields[1:4])))
                if len(vertices[-1]) != 3:
                    raise ValueError(f"{path}:{line_number}: invalid vertex")
            elif fields[0] == "f":
                if len(fields) != 4:
                    raise ValueError(f"{path}:{line_number}: only triangles supported")
                indices = [int(token.split("/", 1)[0]) for token in fields[1:]]
                if 0 in indices:
                    raise ValueError(f"{path}:{line_number}: OBJ index cannot be zero")
                faces.append(
                    tuple(i - 1 if i > 0 else len(vertices) + i for i in indices)
                )
    if not vertices or not faces:
        raise ValueError(f"{path}: no triangle mesh")
    if any(i < 0 or i >= len(vertices) for face in faces for i in face):
        raise ValueError(f"{path}: invalid face index")
    return vertices, faces


def get_sequence(input_path, start):
    """Accept mesh_%04d.obj, or a first file such as mesh_0000.obj."""
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
                name[: match.start()] + (token % i) + name[match.end() :]
            )
    else:
        match = re.search(r"(\d+)(?=\.obj$)", name, re.IGNORECASE)
        if match is None:
            raise ValueError(
                "Pass mesh_%04d.obj or a numbered file such as mesh_0000.obj"
            )
        width = len(match.group())
        number = int(match.group()) if start is None else start

        def make_path(i):
            return path.parent / (
                name[: match.start()] + f"{i:0{width}d}" + name[match.end() :]
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
    parser.add_argument("obj_path", help="First numbered OBJ, or pattern mesh_%04d.obj")
    parser.add_argument(
        "--start", type=int, default=None, help="First file index; pattern default: 0"
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=30)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    paths = get_sequence(args.obj_path, args.start)
    output = (args.output or paths[0].with_suffix(".abc")).expanduser().resolve()
    if output.suffix.lower() != ".abc":
        parser.error("Output must have .abc extension")
    if not bpy.app.build_options.alembic:
        raise RuntimeError("This Blender build does not support Alembic")

    # Use a separate scene; export only the mesh created here.
    scene = bpy.data.scenes.new("OBJSequenceExport")
    bpy.context.window.scene = scene
    scene.render.fps = args.fps
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = len(paths)

    vertices, faces = read_obj(paths[0])
    mesh = bpy.data.meshes.new("SequenceMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Sequence", mesh)
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
        points, triangles = read_obj(path)
        if len(points) != len(vertices) or triangles != faces:
            raise ValueError(f"{path}: vertex count or triangle connectivity changed")
        key = obj.shape_key_add(name=f"Frame_{frame}", from_mix=False)
        key.data.foreach_set("co", array("f", (v for p in points for v in p)))
        key.interpolation = "KEY_LINEAR"
        keys.eval_time = key.frame
        keys.keyframe_insert(data_path="eval_time", frame=frame)
        print(f"Loaded {frame}/{len(paths)}: {path.name}")

    # Samples are exactly at integer frames, so action interpolation between
    # those frames does not affect the exported positions.
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
