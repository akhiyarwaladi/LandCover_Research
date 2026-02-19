"""
Download ForestNet Deforestation Driver Dataset

Downloads the Stanford ForestNet dataset containing 2,756 labeled
Indonesian deforestation images with driver classification.

Drivers:
    - Oil Palm Plantation
    - Timber Plantation
    - Smallholder Agriculture
    - Grassland/Shrub

Usage:
    python scripts/download_forestnet.py

Output:
    data/forestnet/ (extracted dataset)

Reference:
    Irvin, J., et al. (2020). ForestNet: Classifying Drivers of
    Deforestation in Indonesia Using Deep Learning on Satellite Imagery.
    arXiv:2011.05479
"""

import os
import sys
import zipfile
import urllib.request
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ForestNet dataset URL
FORESTNET_URL = 'https://zenodo.org/records/4635787/files/ForestNetDataset.zip'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'forestnet')


def download_file(url, output_path, chunk_size=8192):
    """
    Download a file with progress reporting.

    Args:
        url: Download URL
        output_path: Local save path
        chunk_size: Download chunk size
    """
    print(f"  Downloading from: {url}")
    print(f"  Saving to: {output_path}")

    req = urllib.request.urlopen(url)
    total_size = int(req.headers.get('Content-Length', 0))

    downloaded = 0
    with open(output_path, 'wb') as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total_size > 0:
                pct = (downloaded / total_size) * 100
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)",
                      end='', flush=True)

    print()  # Newline after progress


def main():
    """Download and extract ForestNet dataset."""
    print("=" * 60)
    print("FORESTNET DATASET DOWNLOAD")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    zip_path = os.path.join(OUTPUT_DIR, 'ForestNetDataset.zip')

    # Check if already downloaded
    if os.path.exists(os.path.join(OUTPUT_DIR, 'ForestNetDataset')):
        print("\nForestNet dataset already exists. Skipping download.")
        print(f"  Location: {OUTPUT_DIR}")
        return

    # Download
    if not os.path.exists(zip_path):
        print("\nDownloading ForestNet dataset...")
        print("  This may take several minutes depending on connection speed.")
        try:
            download_file(FORESTNET_URL, zip_path)
            print("  Download complete!")
        except Exception as e:
            print(f"\n  ERROR: Download failed: {e}")
            print("\n  Manual download instructions:")
            print(f"  1. Visit: {FORESTNET_URL}")
            print(f"  2. Download ForestNetDataset.zip")
            print(f"  3. Place it in: {OUTPUT_DIR}")
            print(f"  4. Re-run this script to extract")
            return
    else:
        print(f"\nZip file already exists: {zip_path}")

    # Extract
    print("\nExtracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(OUTPUT_DIR)
        print("  Extraction complete!")

        # Clean up zip
        os.remove(zip_path)
        print("  Cleaned up zip file")

    except Exception as e:
        print(f"  ERROR: Extraction failed: {e}")
        return

    # Verify
    print("\nVerifying dataset...")
    dataset_dir = os.path.join(OUTPUT_DIR, 'ForestNetDataset')
    if os.path.exists(dataset_dir):
        items = os.listdir(dataset_dir)
        print(f"  Contents: {len(items)} items")
        for item in sorted(items)[:10]:
            print(f"    {item}")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more")
    else:
        print("  WARNING: Expected directory not found after extraction")

    print("\n" + "=" * 60)
    print("FORESTNET DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Location: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
