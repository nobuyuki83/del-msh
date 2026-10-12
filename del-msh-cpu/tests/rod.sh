#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$CRATE_DIR/.." && pwd)"

# Run the test
(cd "$CRATE_DIR" && cargo test polyline3::test_shorten::test0 -- --nocapture)

# Find Blender
BLENDER="$("$REPO_DIR/asset/blender_find_exe_path.sh")"

OUT_DIR="$REPO_DIR/target/out_del_msh_cpu"

# Convert the ccd3 OBJ sequence to Alembic
"$BLENDER" -b --python-exit-code 1 \
    --python "$REPO_DIR/asset/blender_obj2abc_trimesh3.py" -- \
    "$OUT_DIR/ccd3_%04d.obj" \
    --start 0 \
    --output "$OUT_DIR/ccd3.abc" \
    --fps 30

# Remove the per-frame OBJ files now that the Alembic is written
rm -f "$OUT_DIR"/ccd3_*.obj
