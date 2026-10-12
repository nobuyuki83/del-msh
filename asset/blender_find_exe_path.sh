#!/usr/bin/env bash
set -euo pipefail

blender_path="${BLENDER:-}"

if [[ -z "$blender_path" ]]; then
    blender_path="$(command -v blender || true)"
fi

if [[ -z "$blender_path" ]]; then
    for candidate in \
        "/Applications/Blender.app/Contents/MacOS/Blender" \
        "$HOME/Applications/Blender.app/Contents/MacOS/Blender"
    do
        if [[ -x "$candidate" ]]; then
            blender_path="$candidate"
            break
        fi
    done
fi

if [[ -z "$blender_path" || ! -x "$blender_path" || -d "$blender_path" ]]; then
    echo "Blenderが見つかりません。BLENDERに実行ファイルのパスを指定してください。" >&2
    exit 1
fi

printf '%s\n' "$blender_path"