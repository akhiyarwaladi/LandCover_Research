#!/usr/bin/env python3
"""
Download Data Penutupan Lahan KLHK 2024 untuk Provinsi Jambi
Sumber: https://geoportal.menlhk.go.id/
"""

import requests
import json
import time
import os
import urllib3

# Disable SSL warnings (KLHK server has certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_data_klhk():
    # Konfigurasi URL dan Parameter
    base_url = "https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query"

    # Filter khusus untuk Provinsi Jambi (KODE_PROV = 15)
    # Note: KODE_PROV is SmallInteger, not string!
    where_clause = "KODE_PROV=15"

    output_filename = "KLHK_PL2024_Jambi_Full.geojson"

    # Header user-agent agar tidak dianggap bot berbahaya
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # ---------------------------------------------------------
    # TAHAP 1: Cek Total Data Terlebih Dahulu
    # ---------------------------------------------------------
    print("=" * 60)
    print("DOWNLOAD DATA KLHK - PENUTUPAN LAHAN 2024 JAMBI")
    print("=" * 60)
    print("\n[1/3] Mengecek total data untuk KODE_PROV=15...")

    count_params = {
        "where": where_clause,
        "returnCountOnly": "true",
        "f": "json"
    }

    try:
        r_count = requests.get(base_url, params=count_params, headers=headers, timeout=30, verify=False)
        r_count.raise_for_status()
        total_count = r_count.json().get("count", 0)
        print(f"      Total data ditemukan: {total_count} records")

        if total_count == 0:
            print("      Tidak ada data ditemukan. Script berhenti.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"      Gagal mengecek jumlah data: {e}")
        return None

    # ---------------------------------------------------------
    # TAHAP 2: Looping Download (Paging)
    # ---------------------------------------------------------
    all_features = []
    offset = 0
    step = 1000  # Batas maksimal server

    print(f"\n[2/3] Memulai download (per {step} records)...")
    print("-" * 60)

    start_time = time.time()

    while True:
        # Parameter query utama
        params = {
            "where": where_clause,
            "outFields": "*",
            "returnGeometry": "true",  # PENTING: harus eksplisit minta geometry
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": step,
            "geometryPrecision": 6
        }

        try:
            progress = min(offset + step, total_count)
            print(f"      Downloading: {offset:,} - {progress:,} / {total_count:,} ({(progress/total_count)*100:.1f}%)")

            response = requests.get(base_url, params=params, headers=headers, timeout=60, verify=False)

            if response.status_code != 200:
                print(f"      Error Server: {response.status_code}")
                break

            data = response.json()
            features = data.get("features", [])

            if not features:
                print("      Tidak ada data lagi.")
                break

            all_features.extend(features)

            # Jika data kurang dari limit, ini halaman terakhir
            if len(features) < step:
                break

            offset += step

            # Jeda agar tidak membebani server
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"      Error saat download: {e}")
            print(f"      Mencoba lagi dalam 5 detik...")
            time.sleep(5)
            continue
        except json.JSONDecodeError as e:
            print(f"      Error parsing JSON: {e}")
            break

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"      Download selesai dalam {elapsed:.1f} detik")
    print(f"      Total features: {len(all_features):,}")

    # ---------------------------------------------------------
    # TAHAP 3: Simpan ke File
    # ---------------------------------------------------------
    print(f"\n[3/3] Menyimpan ke file...")

    if all_features:
        final_geojson = {
            "type": "FeatureCollection",
            "name": "PL_2024_Jambi",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": all_features
        }

        with open(output_filename, "w") as f:
            json.dump(final_geojson, f)

        file_size = os.path.getsize(output_filename) / (1024 * 1024)
        print(f"      File tersimpan: {output_filename}")
        print(f"      Ukuran file: {file_size:.2f} MB")

        return output_filename
    else:
        print("      Tidak ada data untuk disimpan.")
        return None


def analyze_geojson(filename):
    """Analisis hasil download"""
    print("\n" + "=" * 60)
    print("ANALISIS DATA")
    print("=" * 60)

    with open(filename, 'r') as f:
        data = json.load(f)

    features = data.get('features', [])
    print(f"\nTotal polygon: {len(features):,}")

    # KLHK codes reference
    klhk_codes = {
        2001: 'Hutan Lahan Kering Primer',
        2002: 'Hutan Lahan Kering Sekunder',
        2003: 'Hutan Rawa Primer',
        2004: 'Hutan Rawa Sekunder',
        2005: 'Hutan Mangrove Primer',
        2006: 'Hutan Mangrove Sekunder',
        2007: 'Hutan Tanaman',
        2010: 'Perkebunan',
        2012: 'Pemukiman',
        2014: 'Tanah Terbuka',
        3000: 'Savana/Padang Rumput',
        5001: 'Tubuh Air',
        2500: 'Semak Belukar',
        20041: 'Belukar Rawa',
        20051: 'Pertanian Lahan Kering',
        20091: 'Pertanian Lahan Kering Campur',
        20092: 'Sawah',
        20093: 'Tambak',
        20094: 'Rawa',
    }

    # Count by class
    from collections import Counter

    # Try different field names
    code_field = None
    sample_props = features[0].get('properties', {}) if features else {}

    for field in ['PL2024_ID', 'PL2024', 'KELAS', 'ID_PL']:
        if field in sample_props:
            code_field = field
            break

    if not code_field:
        print(f"\nAvailable fields: {list(sample_props.keys())}")
        # Use first numeric-looking field
        for k, v in sample_props.items():
            if isinstance(v, (int, float)) and v > 1000:
                code_field = k
                break

    if code_field:
        codes = [f['properties'].get(code_field) for f in features]
        counter = Counter(codes)

        print(f"\nDistribusi Kelas (field: {code_field}):")
        print("-" * 60)
        print(f"{'Kode':<8} {'Nama':<35} {'Jumlah':<10} {'%':<6}")
        print("-" * 60)

        total = sum(counter.values())
        for code in sorted(counter.keys()):
            if code is not None:
                name = klhk_codes.get(int(code), f'Unknown')
                count = counter[code]
                pct = (count / total) * 100
                print(f"{code:<8} {name:<35} {count:<10} {pct:.1f}%")

        print("-" * 60)
        print(f"{'TOTAL':<8} {'':<35} {total:<10} 100.0%")
    else:
        print("Tidak dapat menemukan field kode tutupan lahan")
        print(f"Fields tersedia: {list(sample_props.keys())}")


if __name__ == "__main__":
    # Download data
    result = download_data_klhk()

    # Analyze if successful
    if result:
        analyze_geojson(result)
        print("\n" + "=" * 60)
        print("SELESAI!")
        print("=" * 60)
