#!/usr/bin/env python3
"""
Download KLHK data using WHERE clause partitioning
Since resultOffset doesn't work with KMZ, partition by OBJECTID ranges
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
OUTPUT_DIR = "data/klhk/partitions"
BATCH_SIZE = 1000

# We know the OBJECTID range from the attributes download
# From KLHK_PL2024_Jambi_Full.geojson: 28,100 total records
# We need to partition by OBJECTID ranges

def get_objectid_ranges():
    """Get OBJECTID ranges from the full attributes file"""
    print("Loading OBJECTID ranges from full attributes file...")

    # Load the attributes-only file we downloaded earlier
    attr_file = 'data/klhk/KLHK_PL2024_Jambi_Full.geojson'

    if os.path.exists(attr_file):
        gdf = gpd.read_file(attr_file)
        objectids = sorted(gdf['OBJECTID'].astype(int).tolist())

        print(f"  Total records: {len(objectids):,}")
        print(f"  OBJECTID range: {objectids[0]} - {objectids[-1]}")

        # Create partitions of ~1000 OBJECTIDs each
        partitions = []
        for i in range(0, len(objectids), BATCH_SIZE):
            batch_ids = objectids[i:i+BATCH_SIZE]
            min_id = batch_ids[0]
            max_id = batch_ids[-1]
            partitions.append((min_id, max_id, len(batch_ids)))

        print(f"  Created {len(partitions)} partitions")
        return partitions
    else:
        print(f"  ❌ Attributes file not found: {attr_file}")
        print("  Using estimated ranges instead...")

        # Fallback: use estimated ranges
        # Estimate based on 28,100 records
        partitions = []
        for i in range(0, 30):
            min_id = i * 1000
            max_id = min_id + 1000
            partitions.append((min_id, max_id, 1000))

        return partitions

def download_partition(min_oid, max_oid, partition_num):
    """Download one partition using OBJECTID WHERE clause"""

    # WHERE clause with OBJECTID range
    where_clause = f"KODE_PROV=15 AND OBJECTID>={min_oid} AND OBJECTID<={max_oid}"

    params = {
        'where': where_clause,
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'kmz'
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    kmz_file = f"{OUTPUT_DIR}/partition_{partition_num:03d}_oid_{min_oid}_{max_oid}.kmz"

    print(f"\n[Partition {partition_num}] OBJECTID {min_oid}-{max_oid}...")
    print(f"  WHERE: {where_clause}")

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

def convert_kmz_to_geojson(kmz_file, partition_num):
    """Convert KMZ to GeoJSON"""
    kml_file = kmz_file.replace('.kmz', '.kml')
    geojson_file = kmz_file.replace('.kmz', '_raw.geojson')

    print(f"  Converting to GeoJSON...")

    try:
        # Extract KMZ
        import zipfile
        with zipfile.ZipFile(kmz_file, 'r') as zip_ref:
            # Extract to temp location
            temp_dir = os.path.join(OUTPUT_DIR, f'temp_{partition_num}')
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)

        # Find and rename KML
        kml_extracted = os.path.join(temp_dir, 'doc.kml')
        if os.path.exists(kml_extracted):
            # Move to partition directory
            import shutil
            shutil.move(kml_extracted, kml_file)
            # Clean up temp dir
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Convert to GeoJSON
        cmd = f'ogr2ogr -f GeoJSON "{geojson_file}" "{kml_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0 and os.path.exists(geojson_file):
            print(f"  ✅ Converted to GeoJSON")
            return geojson_file
        else:
            print(f"  ❌ Conversion failed")
            return None

    except Exception as e:
        print(f"  ❌ Conversion error: {e}")
        return None

def parse_description(desc):
    """Parse HTML description"""
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
                data[cells[0].get_text(strip=True)] = cells[1].get_text(strip=True)
        return data
    except:
        return {}

def clean_geojson(raw_file, partition_num):
    """Parse and clean GeoJSON"""
    clean_file = raw_file.replace('_raw.geojson', '_clean.geojson')

    print(f"  Parsing attributes...")

    try:
        gdf = gpd.read_file(raw_file)

        parsed_data = []
        for idx, row in gdf.iterrows():
            attrs = parse_description(row.get('description', ''))
            attrs['geometry'] = row['geometry']
            parsed_data.append(attrs)

        gdf_clean = gpd.GeoDataFrame(parsed_data, crs=gdf.crs)
        gdf_clean.to_file(clean_file, driver='GeoJSON')

        print(f"  ✅ Cleaned {len(gdf_clean)} features")
        return clean_file, len(gdf_clean)

    except Exception as e:
        print(f"  ❌ Cleaning error: {e}")
        return None, 0

def merge_partitions(output_file):
    """Merge all partitions"""
    print(f"\n{'='*60}")
    print("MERGING PARTITIONS")
    print("="*60)

    clean_files = sorted([
        os.path.join(OUTPUT_DIR, f)
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith('_clean.geojson')
    ])

    print(f"Found {len(clean_files)} partition files")

    if not clean_files:
        print("❌ No files to merge!")
        return None

    gdfs = []
    total = 0

    for pfile in clean_files:
        print(f"  Loading {os.path.basename(pfile)}...", end='')
        gdf = gpd.read_file(pfile)
        gdfs.append(gdf)
        total += len(gdf)
        print(f" {len(gdf)} features")

    print(f"\nConcatenating {total:,} features...")
    gdf_merged = pd.concat(gdfs, ignore_index=True)
    gdf_merged = gpd.GeoDataFrame(gdf_merged, crs=gdfs[0].crs)

    # Remove duplicates
    if 'OBJECTID' in gdf_merged.columns:
        before = len(gdf_merged)
        gdf_merged = gdf_merged.drop_duplicates(subset='OBJECTID', keep='first')
        after = len(gdf_merged)
        if before != after:
            print(f"  Removed {before - after} duplicates")

    print(f"\nSaving to {output_file}...")
    gdf_merged.to_file(output_file, driver='GeoJSON')

    print(f"\n✅ MERGED FILE CREATED!")
    print(f"   Total features: {len(gdf_merged):,}")
    print(f"   Output: {output_file}")

    return gdf_merged

if __name__ == "__main__":
    print("="*60)
    print("KLHK PARTITIONED DOWNLOAD - OBJECTID WHERE CLAUSE METHOD")
    print("="*60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get OBJECTID ranges
    partitions = get_objectid_ranges()

    print(f"\nStarting download of {len(partitions)} partitions...")

    successful = 0
    failed = 0
    total_features = 0

    for i, (min_oid, max_oid, expected_count) in enumerate(partitions, 1):
        # Download
        kmz = download_partition(min_oid, max_oid, i)
        if not kmz:
            failed += 1
            continue

        # Convert
        raw_geojson = convert_kmz_to_geojson(kmz, i)
        if not raw_geojson:
            failed += 1
            continue

        # Clean
        clean_file, count = clean_geojson(raw_geojson, i)
        if not clean_file:
            failed += 1
            continue

        successful += 1
        total_features += count
        print(f"  Progress: {total_features:,} features")

        time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total features: {total_features:,}")

    # Merge
    if successful > 0:
        merge_partitions("data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson")
