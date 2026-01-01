#!/usr/bin/env python3
"""
Try downloading older KLHK data (2023, 2022, 2021)
Older endpoints might still have geometry access enabled
"""

import requests
import json
import urllib3
urllib3.disable_warnings()

# URLs untuk coba (berbagai tahun)
KLHK_URLS = {
    '2024': 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query',
    '2023': 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2023/MapServer/0/query',
    '2022': 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2022/MapServer/0/query',
    '2021': 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2021/MapServer/0/query',
    '2020': 'https://geoportal.menlhk.go.id/server/rest/services/KLHK/Penutupan_Lahan_Tahun_2020/MapServer/0/query',
    '2019': 'https://geoportal.menlhk.go.id/server/rest/services/KLHK/Penutupan_Lahan_Tahun_2019/MapServer/0/query',
}

def test_geometry_access(url, year):
    """Test if geometry is accessible for a given year"""
    print(f"\n{'='*60}")
    print(f"Testing {year} data...")
    print(f"URL: {url}")
    print('='*60)

    params = {
        'where': 'KODE_PROV=15',  # Jambi
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'geojson',
        'resultRecordCount': 1
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, params=params, headers=headers, verify=False, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if 'features' in data and len(data['features']) > 0:
                feature = data['features'][0]
                geometry = feature.get('geometry')

                print(f"✅ URL accessible!")
                print(f"   Features found: {len(data['features'])}")

                if geometry and geometry != 'null':
                    print(f"   🎉 GEOMETRY EXISTS!")
                    print(f"   Geometry type: {geometry.get('type') if isinstance(geometry, dict) else type(geometry)}")

                    if isinstance(geometry, dict):
                        if 'coordinates' in geometry:
                            print(f"   Coordinates: {len(geometry['coordinates'])} elements")
                        elif 'rings' in geometry:
                            print(f"   Rings: {len(geometry['rings'])} rings")

                    return True, url
                else:
                    print(f"   ❌ Geometry is NULL")
                    return False, None
            else:
                print(f"   ⚠️  No features found")
                return False, None
        else:
            print(f"   ❌ HTTP {response.status_code}")
            return False, None

    except requests.exceptions.Timeout:
        print(f"   ⏱️  Timeout")
        return False, None
    except requests.exceptions.ConnectionError:
        print(f"   🔌 Connection failed")
        return False, None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False, None


def download_full_data(url, year, output_file):
    """Download full dataset if geometry is available"""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING FULL {year} DATA")
    print('='*60)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    # First, get count
    count_params = {
        'where': 'KODE_PROV=15',
        'returnCountOnly': 'true',
        'f': 'json'
    }

    try:
        r = requests.get(url, params=count_params, headers=headers, verify=False, timeout=30)
        total_count = r.json().get('count', 0)
        print(f"Total records: {total_count:,}")

        if total_count == 0:
            print("No data found!")
            return None

        # Download in batches
        all_features = []
        offset = 0
        step = 1000

        while offset < total_count:
            params = {
                'where': 'KODE_PROV=15',
                'outFields': '*',
                'returnGeometry': 'true',
                'f': 'geojson',
                'resultOffset': offset,
                'resultRecordCount': step,
                'geometryPrecision': 6
            }

            progress = min(offset + step, total_count)
            print(f"Downloading: {offset:,} - {progress:,} / {total_count:,} ({(progress/total_count)*100:.1f}%)")

            response = requests.get(url, params=params, headers=headers, verify=False, timeout=60)

            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])

                if not features:
                    break

                all_features.extend(features)

                if len(features) < step:
                    break
            else:
                print(f"Error: HTTP {response.status_code}")
                break

            offset += step

        print(f"\nTotal features downloaded: {len(all_features):,}")

        # Save to file
        geojson = {
            'type': 'FeatureCollection',
            'name': f'KLHK_PL{year}_Jambi',
            'features': all_features
        }

        with open(output_file, 'w') as f:
            json.dump(geojson, f)

        print(f"✅ Saved to: {output_file}")

        # Verify geometry
        has_geometry = False
        for feat in all_features[:10]:
            if feat.get('geometry') and feat.get('geometry') != 'null':
                has_geometry = True
                break

        if has_geometry:
            print(f"✅ GEOMETRY CONFIRMED in downloaded data!")
        else:
            print(f"⚠️  WARNING: Downloaded data has NULL geometry")

        return output_file

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    import os

    print("\n" + "="*60)
    print("KLHK OLDER DATA GEOMETRY TEST")
    print("="*60)
    print("\nTesting which KLHK years still have geometry access...\n")

    results = {}

    # Test all years
    for year, url in KLHK_URLS.items():
        has_geom, working_url = test_geometry_access(url, year)
        results[year] = {
            'has_geometry': has_geom,
            'url': working_url
        }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    working_years = []
    for year, result in results.items():
        status = "✅ HAS GEOMETRY!" if result['has_geometry'] else "❌ No geometry"
        print(f"{year}: {status}")

        if result['has_geometry']:
            working_years.append(year)

    # Download if found
    if working_years:
        print(f"\n🎉 Found {len(working_years)} year(s) with geometry!")
        print(f"Years: {', '.join(working_years)}")

        # Download the most recent one
        latest_year = max(working_years)
        print(f"\n📥 Downloading {latest_year} data...")

        output_dir = 'data/klhk'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/KLHK_PL{latest_year}_Jambi_WithGeometry.geojson'

        url = results[latest_year]['url']
        download_full_data(url, latest_year, output_file)

    else:
        print("\n😞 No years found with geometry access")
        print("\nRECOMMENDATIONS:")
        print("1. Email KLHK: geoportal@menlhk.go.id")
        print("2. Try manual WebGIS download")
        print("3. Consider LapakGIS (paid)")
        print("\nSee: docs/GET_KLHK_GEOMETRY.md for detailed guide")
