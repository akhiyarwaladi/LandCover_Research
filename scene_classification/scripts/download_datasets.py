"""
Download Scene Classification Benchmark Datasets

Downloads EuroSAT (automatic) and provides instructions for NWPU-RESISC45 and AID.

Usage:
    python scripts/download_datasets.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.dataset_loader import (
    download_eurosat, download_nwpu_resisc45, download_aid, verify_dataset
)


def main():
    print("=" * 60)
    print("DOWNLOAD SCENE CLASSIFICATION DATASETS")
    print("=" * 60)

    # EuroSAT (automatic)
    print("\n[1/3] EuroSAT")
    try:
        download_eurosat()
    except Exception as e:
        print(f"  Error: {e}")
        print("  Manual download: https://zenodo.org/records/7711810")

    # NWPU-RESISC45 (manual)
    print("\n[2/3] NWPU-RESISC45")
    download_nwpu_resisc45()

    # AID (manual)
    print("\n[3/3] AID")
    download_aid()

    # Verify
    print("\n" + "=" * 60)
    print("DATASET STATUS")
    print("=" * 60)
    for name in ['eurosat', 'nwpu_resisc45', 'aid']:
        verify_dataset(name)

    print("\nAfter manual downloads, re-run this script to verify.")


if __name__ == '__main__':
    main()
