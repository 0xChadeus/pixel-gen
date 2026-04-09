"""Download popular palettes from Lospec and save as GIMP .gpl files."""

import argparse
import os
from pathlib import Path

import requests


POPULAR_PALETTES = [
    "pico-8",
    "endesga-32",
    "endesga-64",
    "resurrect-64",
    "lospec500",
    "apollo",
    "sweetie-16",
    "bubblegum-16",
    "oil-6",
    "slso8",
    "zughy-32",
    "fantasy-24",
    "aap-64",
    "journey",
    "na16",
    "cc-29",
    "japanese-woodblock",
    "duel",
    "blessing",
    "nyx8",
]


def download_palette(slug: str, output_dir: str) -> bool:
    """Download a palette from Lospec as .hex and convert to .gpl.

    Args:
        slug: Lospec palette slug (e.g., "pico-8")
        output_dir: Directory to save .gpl file

    Returns:
        True if successful
    """
    url = f"https://lospec.com/palette-list/{slug}.hex"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Failed to download {slug}: {e}")
        return False

    hex_colors = [line.strip() for line in resp.text.strip().split("\n") if line.strip()]

    if not hex_colors:
        print(f"  No colors found for {slug}")
        return False

    # Write as GIMP .gpl
    gpl_path = Path(output_dir) / f"{slug}.gpl"
    with open(gpl_path, "w") as f:
        f.write("GIMP Palette\n")
        f.write(f"Name: {slug}\n")
        f.write(f"Columns: {min(len(hex_colors), 16)}\n")
        f.write("#\n")
        for h in hex_colors:
            h = h.lstrip("#")
            r = int(h[0:2], 16)
            g = int(h[2:4], 16)
            b = int(h[4:6], 16)
            f.write(f"{r:3d} {g:3d} {b:3d}\t{h}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download Lospec palettes")
    parser.add_argument("--output", default="aseprite_plugin/palettes",
                        help="Output directory for .gpl files")
    parser.add_argument("--palettes", nargs="+", default=POPULAR_PALETTES)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    success = 0
    for slug in args.palettes:
        print(f"Downloading {slug}...")
        if download_palette(slug, args.output):
            success += 1

    print(f"\nDownloaded {success}/{len(args.palettes)} palettes to {args.output}")


if __name__ == "__main__":
    main()
