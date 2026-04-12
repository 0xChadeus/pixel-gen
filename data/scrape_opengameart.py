"""Scrape pixel art from OpenGameArt.org (CC0 / CC-BY licensed).

Respects rate limits with delays between requests. Downloads individual
images (not archives) tagged with pixel art.

Usage:
    python -m data.scrape_opengameart --output data/raw/opengameart --max-pages 50
"""

import argparse
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from tqdm import tqdm


BASE_URL = "https://opengameart.org"
SEARCH_URL = BASE_URL + "/art-search-advanced"
ALLOWED_LICENSES = {"CC0", "CC-BY 3.0", "CC-BY 4.0", "CC-BY-SA 3.0", "CC-BY-SA 4.0"}
REQUEST_DELAY = 2.0  # seconds between requests to be polite


def _get_page(session: requests.Session, page: int) -> str:
    """Fetch a search results page for pixel art assets."""
    params = {
        "keys": "",
        "field_art_type_tid[]": "9",  # 2D Art
        "field_art_tags_tid": "pixel art",
        "sort_by": "count",  # sort by popularity
        "sort_order": "DESC",
        "page": page,
    }
    resp = session.get(SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def _extract_asset_links(html: str) -> list[str]:
    """Extract asset page links from search results HTML."""
    pattern = r'href="(/content/[^"]+)"'
    matches = re.findall(pattern, html)
    # Deduplicate while preserving order
    seen = set()
    links = []
    for m in matches:
        if m not in seen and "/content/faq" not in m:
            seen.add(m)
            links.append(m)
    return links


def _extract_image_urls(html: str) -> list[str]:
    """Extract direct image download URLs from an asset page."""
    # Look for preview images and direct file links
    patterns = [
        r'href="(https://opengameart\.org/sites/default/files/[^"]+\.png)"',
        r'src="(https://opengameart\.org/sites/default/files/[^"]+\.png)"',
    ]
    urls = set()
    for pattern in patterns:
        urls.update(re.findall(pattern, html))

    # Filter out thumbnails and icons
    filtered = []
    for url in urls:
        lower = url.lower()
        if "thumbnail" in lower or "icon" in lower or "styles/" in lower:
            continue
        filtered.append(url)
    return filtered


def _check_license(html: str) -> bool:
    """Check if the asset has an allowed license."""
    for lic in ALLOWED_LICENSES:
        if lic in html:
            return True
    return False


def scrape(output_dir: str, max_pages: int = 50, max_images: int = 0):
    """Scrape pixel art images from OpenGameArt."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "PixelGenTrainingDataCollector/1.0 (research; polite scraping)",
    })

    downloaded = 0
    total_assets = 0

    for page in tqdm(range(max_pages), desc="Pages"):
        try:
            html = _get_page(session, page)
        except Exception as e:
            print(f"Failed to fetch page {page}: {e}")
            break

        asset_links = _extract_asset_links(html)
        if not asset_links:
            print(f"No more assets found at page {page}")
            break

        for link in asset_links:
            if max_images > 0 and downloaded >= max_images:
                return downloaded

            time.sleep(REQUEST_DELAY)
            asset_url = urljoin(BASE_URL, link)

            try:
                resp = session.get(asset_url, timeout=30)
                resp.raise_for_status()
                asset_html = resp.text
            except Exception:
                continue

            if not _check_license(asset_html):
                continue

            image_urls = _extract_image_urls(asset_html)
            total_assets += 1

            for img_url in image_urls:
                try:
                    time.sleep(REQUEST_DELAY)
                    img_resp = session.get(img_url, timeout=60)
                    img_resp.raise_for_status()

                    # Generate filename from URL
                    fname = img_url.split("/")[-1]
                    fname = re.sub(r'[^\w\-.]', '_', fname)
                    save_path = out / fname

                    if save_path.exists():
                        continue

                    save_path.write_bytes(img_resp.content)
                    downloaded += 1

                    if max_images > 0 and downloaded >= max_images:
                        print(f"\nReached max images ({max_images})")
                        return downloaded

                except Exception:
                    continue

        print(f"Page {page}: {downloaded} images downloaded from {total_assets} assets")

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Scrape pixel art from OpenGameArt")
    parser.add_argument("--output", default="data/raw/opengameart",
                        help="Output directory")
    parser.add_argument("--max-pages", type=int, default=50,
                        help="Maximum search pages to crawl")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Maximum images to download (0=unlimited)")
    args = parser.parse_args()

    count = scrape(args.output, args.max_pages, args.max_images)
    print(f"\nDone. Downloaded {count} images to {args.output}")


if __name__ == "__main__":
    main()
