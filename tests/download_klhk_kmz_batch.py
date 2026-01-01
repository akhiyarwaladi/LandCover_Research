#!/usr/bin/env python3
"""
Automated batch download of KLHK data using KMZ export
Downloads all 28,100 Jambi polygons with geometry in batches of 1,000
"""

import requests
import subprocess
import os
import geopandas as gpd
import pandas as pd
from bs4 import BeautifulSoup
import time
import urllib3
urllib3.disable_warnings()

# Configuration
BASE_URL = "https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query"
OUTPUT_DIR = "data/klhk/batches"
BATCH_SIZE = 1000
TOTAL_RECORDS = 28100  # From previous full attribute download

def download_kmz_batch(offset, batch_num):
    """Download one batch as KMZ"""
    params = {
        'where': 'KODE_PROV=15',
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'kmz',
        'resultOffset': offset,
        'resultRecordCount': BATCH_SIZE
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    kmz_file = f"{OUTPUT_DIR}/batch_{batch_num:03d}_offset_{offset}.kmz"

    print(f"\n[Batch {batch_num}] Downloading offset {offset}...")
    print(f"  URL: {BASE_URL}")
    print(f"  Params: resultOffset={offset}, resultRecordCount={BATCH_SIZE}")

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            headers=headers,
            verify=False,
            timeout=120
        )

        if response.status_code == 200:
            with open(kmz_file, 'wb') as f:
                f.write(response.content)

            file_size = os.path.getsize(kmz_file)
            print(f"  ✅ Downloaded: {file_size:,} bytes")

            # Check if empty (very small file = no data)
            if file_size < 500:
                print(f"  ⚠️  Warning: File too small, may be empty")
                return None

            return kmz_file
        else:
            print(f"  ❌ HTTP {response.status_code}")
            return None

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def convert_kmz_to_geojson(kmz_file, batch_num):
    """Convert KMZ to GeoJSON using ogr2ogr"""
    kml_file = kmz_file.replace('.kmz', '.kml')
    geojson_file = kmz_file.replace('.kmz', '_raw.geojson')

    print(f"  Converting to GeoJSON...")

    try:
        # Extract KMZ to KML
        import zipfile
        with zipfile.ZipFile(kmz_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(kmz_file))

        # Find extracted KML (usually doc.kml)
        kml_extracted = os.path.join(os.path.dirname(kmz_file), 'doc.kml')
        if os.path.exists(kml_extracted):
            os.rename(kml_extracted, kml_file)

        # Convert KML to GeoJSON
        cmd = f'ogr2ogr -f GeoJSON "{geojson_file}" "{kml_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(geojson_file):
            print(f"  ✅ Converted to GeoJSON")
            return geojson_file
        else:
            print(f"  ❌ Conversion failed: {result.stderr}")
            return None

    except Exception as e:
        print(f"  ❌ Conversion error: {e}")
        return None

def parse_description(desc):
    """Parse HTML description table to dictionary"""
    if pd.isna(desc) or not desc:
        return {}

    try:
        soup = BeautifulSoup(desc, 'html.parser')
        table = soup.find('table')
        if not table:
            return {}

        data = {}
        rows = table.find_all('tr')
        for row in rows[1:]:
            cells = row.find_all('td')
            if len(cells) == 2:
                field_name = cells[0].get_text(strip=True)
                field_value = cells[1].get_text(strip=True)
                data[field_name] = field_value
        return data
    except:
        return {}

def clean_geojson(raw_geojson_file, batch_num):
    """Parse descriptions and create clean GeoJSON"""
    clean_file = raw_geojson_file.replace('_raw.geojson', '_clean.geojson')

    print(f"  Parsing attributes...")

    try:
        gdf = gpd.read_file(raw_geojson_file)

        # Parse descriptions
        parsed_data = []
        for idx, row in gdf.iterrows():
            attrs = parse_description(row.get('description', ''))
            attrs['geometry'] = row['geometry']
            parsed_data.append(attrs)

        gdf_clean = gpd.GeoDataFrame(parsed_data, crs=gdf.crs)

        # Save
        gdf_clean.to_file(clean_file, driver='GeoJSON')

        feature_count = len(gdf_clean)
        print(f"  ✅ Cleaned {feature_count} features")

        return clean_file, feature_count

    except Exception as e:
        print(f"  ❌ Cleaning error: {e}")
        return None, 0

def merge_all_batches(output_file):
    """Merge all batch GeoJSON files into one"""
    print(f"\n{'='*60}")
    print(f"MERGING ALL BATCHES")
    print(f"{'='*60}")

    # Find all clean GeoJSON files
    clean_files = sorted([
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith('_clean.geojson')
    ])

    print(f"Found {len(clean_files)} batch files")

    if not clean_files:
        print("❌ No batch files found!")
        return None

    # Load and concatenate all
    gdfs = []
    total_features = 0

    for batch_file in clean_files:
        print(f"  Loading {os.path.basename(batch_file)}...", end='')
        gdf = gpd.read_file(batch_file)
        gdfs.append(gdf)
        total_features += len(gdf)
        print(f" {len(gdf)} features")

    # Concatenate
    print(f"\nConcatenating {total_features} features...")
    gdf_merged = pd.concat(gdfs, ignore_index=True)
    gdf_merged = gpd.GeoDataFrame(gdf_merged, crs=gdfs[0].crs)

    # Remove duplicates based on OBJECTID
    if 'OBJECTID' in gdf_merged.columns:
        before = len(gdf_merged)
        gdf_merged = gdf_merged.drop_duplicates(subset='OBJECTID', keep='first')
        after = len(gdf_merged)
        if before != after:
            print(f"  Removed {before - after} duplicates")

    # Save merged
    print(f"\nSaving to {output_file}...")
    gdf_merged.to_file(output_file, driver='GeoJSON')

    print(f"\n✅ MERGED FILE CREATED!")
    print(f"   Total features: {len(gdf_merged):,}")
    print(f"   Output: {output_file}")

    # Statistics
    if 'ID Penutupan Lahan Tahun 2024' in gdf_merged.columns:
        print(f"\n--- CLASS DISTRIBUTION ---")
        class_dist = gdf_merged['ID Penutupan Lahan Tahun 2024'].value_counts()
        for class_id, count in class_dist.head(10).items():
            pct = (count / len(gdf_merged)) * 100
            print(f"  {class_id}: {count:,} ({pct:.1f}%)")

    return gdf_merged

if __name__ == "__main__":
    print("="*60)
    print("KLHK BATCH DOWNLOAD - KMZ METHOD")
    print("="*60)
    print(f"Total records: {TOTAL_RECORDS:,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Estimated batches: {(TOTAL_RECORDS // BATCH_SIZE) + 1}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Track progress
    successful_batches = 0
    failed_batches = 0
    total_features = 0

    # Download all batches
    batch_num = 0
    for offset in range(0, TOTAL_RECORDS, BATCH_SIZE):
        batch_num += 1

        # Download KMZ
        kmz_file = download_kmz_batch(offset, batch_num)
        if not kmz_file:
            failed_batches += 1
            continue

        # Convert to GeoJSON
        raw_geojson = convert_kmz_to_geojson(kmz_file, batch_num)
        if not raw_geojson:
            failed_batches += 1
            continue

        # Clean attributes
        clean_file, feature_count = clean_geojson(raw_geojson, batch_num)
        if not clean_file:
            failed_batches += 1
            continue

        successful_batches += 1
        total_features += feature_count

        print(f"  Progress: {total_features:,} / {TOTAL_RECORDS:,} ({(total_features/TOTAL_RECORDS)*100:.1f}%)")

        # Small delay to avoid overwhelming server
        time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {failed_batches}")
    print(f"Total features downloaded: {total_features:,}")

    # Merge all batches
    if successful_batches > 0:
        merged_file = "data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson"
        merge_all_batches(merged_file)
    else:
        print("\n❌ No successful batches to merge")
