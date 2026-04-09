#!/bin/bash
# Symlink the plugin directory into Aseprite's extensions folder.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLUGIN_DIR="$PROJECT_DIR/aseprite_plugin"

# Aseprite config directory varies by OS
if [ -d "$HOME/.config/aseprite/extensions" ]; then
    ASEPRITE_EXT="$HOME/.config/aseprite/extensions"
elif [ -d "$HOME/Library/Application Support/Aseprite/extensions" ]; then
    ASEPRITE_EXT="$HOME/Library/Application Support/Aseprite/extensions"
elif [ -d "$APPDATA/Aseprite/extensions" ]; then
    ASEPRITE_EXT="$APPDATA/Aseprite/extensions"
else
    echo "Could not find Aseprite extensions directory."
    echo "Please symlink manually: ln -s $PLUGIN_DIR <aseprite-extensions>/pixel-gen"
    exit 1
fi

TARGET="$ASEPRITE_EXT/pixel-gen"
if [ -e "$TARGET" ]; then
    echo "Already installed at $TARGET"
else
    ln -s "$PLUGIN_DIR" "$TARGET"
    echo "Installed: $TARGET -> $PLUGIN_DIR"
fi
echo "Restart Aseprite to load the plugin."
