#!/usr/bin/env python3
"""
Test: Does GeoJSON format also return geometry, or only KMZ?
Comparison test untuk lihat mana yang bekerja
"""

import requests
import urllib3
urllib3.disable_warnings()

BASE_URL = "https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query"

print("="*70)
print("COMPARISON TEST: GeoJSON vs KMZ Geometry Access")
print("="*70)

# Test same OBJECTID range with both formats
test_where = "KODE_PROV=15 AND OBJECTID>=182 AND OBJECTID<=200"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Referer': 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query'
}

# TEST 1: GeoJSON format
print("\n" + "="*70)
print("TEST 1: f=geojson")
print("="*70)

params_geojson = {
    'where': test_where,
    'outFields': '*',
    'returnGeometry': 'true',
    'f': 'geojson'
}

print(f"WHERE: {test_where}")
print(f"Format: f=geojson")
print(f"Testing...")

try:
    response = requests.get(BASE_URL, params=params_geojson, headers=headers, verify=False, timeout=30)
    print(f"HTTP Status: {response.status_code}")
    print(f"Response size: {len(response.content):,} bytes")

    if response.status_code == 200:
        data = response.json()
        features = data.get('features', [])
        print(f"Features returned: {len(features)}")

        if len(features) > 0:
            # Check geometry
            has_geometry = 0
            null_geometry = 0

            for feat in features:
                geom = feat.get('geometry')
                if geom and geom != 'null' and geom is not None:
                    has_geometry += 1
                else:
                    null_geometry += 1

            print(f"\nGeometry check:")
            print(f"  ✅ Has geometry: {has_geometry}")
            print(f"  ❌ NULL geometry: {null_geometry}")

            if has_geometry > 0:
                print(f"\n🎉 GeoJSON FORMAT WORKS!")
                # Show sample
                sample = features[0]
                geom = sample.get('geometry')
                if geom and 'coordinates' in geom:
                    print(f"\nSample geometry type: {geom.get('type')}")
                    coords = geom.get('coordinates', [])
                    if coords:
                        coord_count = len(coords[0]) if isinstance(coords[0], list) else len(coords)
                        print(f"Coordinate count: {coord_count}")
                        print(f"First coordinate: {coords[0][0] if isinstance(coords[0], list) else coords[0]}")
            else:
                print(f"\n❌ GeoJSON FORMAT: ALL GEOMETRY NULL!")
    else:
        print(f"❌ Request failed")

except Exception as e:
    print(f"❌ Error: {e}")

# TEST 2: KMZ format (we know this works)
print("\n" + "="*70)
print("TEST 2: f=kmz (untuk perbandingan)")
print("="*70)

params_kmz = {
    'where': test_where,
    'outFields': '*',
    'returnGeometry': 'true',
    'f': 'kmz'
}

print(f"WHERE: {test_where}")
print(f"Format: f=kmz")
print(f"Testing...")

try:
    response = requests.get(BASE_URL, params=params_kmz, headers=headers, verify=False, timeout=30)
    print(f"HTTP Status: {response.status_code}")
    print(f"Response size: {len(response.content):,} bytes")

    if response.status_code == 200:
        # Save temp file
        temp_file = 'temp_test.kmz'
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        # Extract and check
        import zipfile
        import os
        from bs4 import BeautifulSoup

        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall('temp_kmz')

        kml_file = 'temp_kmz/doc.kml'
        if os.path.exists(kml_file):
            with open(kml_file, 'r', encoding='utf-8') as f:
                kml_content = f.read()

            soup = BeautifulSoup(kml_content, 'xml')
            placemarks = soup.find_all('Placemark')

            print(f"Features returned: {len(placemarks)}")

            # Count geometries
            has_geometry = 0
            for pm in placemarks:
                if pm.find('Polygon') or pm.find('coordinates'):
                    has_geometry += 1

            print(f"\nGeometry check:")
            print(f"  ✅ Has geometry: {has_geometry}")
            print(f"\n✅ KMZ FORMAT WORKS (as expected)")

            # Cleanup
            import shutil
            os.remove(temp_file)
            shutil.rmtree('temp_kmz', ignore_errors=True)

except Exception as e:
    print(f"❌ Error: {e}")

# SUMMARY
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("""
Hypothesis sebelumnya:
- GeoJSON format return NULL geometry
- Hanya KMZ format yang bisa download geometry

Test ini akan buktikan:
- Apakah GeoJSON sekarang bisa return geometry?
- Atau tetap hanya KMZ yang berfungsi?

Result: (lihat output di atas)
""")

print("="*70)
