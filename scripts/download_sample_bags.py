#!/usr/bin/env python3
"""
Download Sample RealSense Bag Files

Downloads sample RGB-D recordings for testing without physical camera.
"""

import os
import urllib.request
import sys
from pathlib import Path


SAMPLE_BAGS = [
    {
        'name': 'outdoor_scene',
        'url': 'https://librealsense.intel.com/rs-tests/TestData/outdoors_1color.bag',
        'size': '~50MB',
        'description': 'Outdoor scene with depth'
    },
    {
        'name': 'stairs',
        'url': 'https://librealsense.intel.com/rs-tests/TestData/stairs.bag',
        'size': '~30MB',
        'description': 'Indoor stairs navigation'
    }
]


def download_file(url: str, output_path: str):
    """Download file with progress bar."""
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}% ({downloaded/1024/1024:.1f}MB)', end='')
        sys.stdout.flush()

    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
    print()  # New line after progress


def main():
    print("=" * 60)
    print("RealSense Sample Bag File Downloader")
    print("=" * 60)

    # Create data directory
    data_dir = Path(__file__).parent.parent / 'data' / 'bags'
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownload directory: {data_dir}")
    print(f"\nAvailable samples:")

    for i, bag in enumerate(SAMPLE_BAGS, 1):
        print(f"{i}. {bag['name']}")
        print(f"   {bag['description']}")
        print(f"   Size: {bag['size']}")
        print()

    # Ask user which to download
    print("Enter numbers to download (e.g., '1 2' or 'all'):")
    choice = input("> ").strip().lower()

    if choice == 'all':
        to_download = SAMPLE_BAGS
    else:
        indices = [int(x) - 1 for x in choice.split()]
        to_download = [SAMPLE_BAGS[i] for i in indices if 0 <= i < len(SAMPLE_BAGS)]

    if not to_download:
        print("[ERROR] No valid selection")
        return

    # Download files
    print(f"\nDownloading {len(to_download)} file(s)...\n")

    for bag in to_download:
        output_path = data_dir / f"{bag['name']}.bag"

        if output_path.exists():
            print(f"[SKIP] {bag['name']}.bag already exists")
            continue

        try:
            download_file(bag['url'], str(output_path))
            print(f"[OK] Downloaded {bag['name']}.bag")
        except Exception as e:
            print(f"[ERROR] Failed to download {bag['name']}: {e}")

    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    print("\nUsage:")
    print("  from vision.realsense import BagFilePlayer")
    print(f"  camera = BagFilePlayer('{data_dir}/outdoor_scene.bag')")
    print("  camera.start()")
    print()


if __name__ == "__main__":
    main()
