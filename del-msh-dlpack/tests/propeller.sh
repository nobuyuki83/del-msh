#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$CRATE_DIR/.." && pwd)"

# Run the test
(cd "$CRATE_DIR" && cargo test polyline3::test_shorten::test0 -- --nocapture)

# Find Blender
BLENDER="$("$REPO_DIR/asset/blender_find_exe_path.sh")"

OUT_DIR="$REPO_DIR/target/out_dlpack"

TARGET_DIR="$REPO_DIR/target/out_dlpack"

# Convert point-cloud sequence to Alembic (each point rendered as a sphere)
"$BLENDER" -b --python-exit-code 1 \
    --python "$REPO_DIR/asset/blender_obj2abc_vtx2xyz.py" -- \
    "$TARGET_DIR/test_edgegrad_depth_pnt_0.obj" \
    --output "$TARGET_DIR/test_edgegrad_depth_pnt.abc" \
    --fps 30 \
    --radius 0.5
# rm -f "$TARGET_DIR"/test_edgegrad_depth_pnt_*.obj

# Convert boundary mesh sequence to Alembic
"$BLENDER" -b --python-exit-code 1 \
    --python "$REPO_DIR/asset/blender_obj2abc_trimesh3.py" -- \
    "$TARGET_DIR/test_edgegrad_depth_bdy_0.obj" \
    --output "$TARGET_DIR/test_edgegrad_depth_bdy.abc" \
    --fps 30
# rm -f "$TARGET_DIR"/test_edgegrad_depth_bdy_*.obj
