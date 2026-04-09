#!/bin/bash
# Package the Aseprite plugin as a .aseprite-extension file (zip archive).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLUGIN_DIR="$PROJECT_DIR/aseprite_plugin"
OUT="$PROJECT_DIR/pixel-gen.aseprite-extension"

cd "$PLUGIN_DIR"
rm -f "$OUT"
zip -r "$OUT" package.json plugin.lua json.lua sprite_utils.lua palettes/
echo "Built: $OUT"
echo "Install: double-click the file or use Edit > Preferences > Extensions > Add Extension"
