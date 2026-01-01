import requests
import json
import urllib3
urllib3.disable_warnings()

# Check service capabilities
service_url = 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0'

print("Checking layer metadata...")
r = requests.get(f"{service_url}?f=json", verify=False, timeout=30)
metadata = r.json()

print("\nLayer Name:", metadata.get('name'))
print("Geometry Type:", metadata.get('geometryType'))
print("Has Geometry:", metadata.get('hasGeometryProperties'))
print("Supports Query:", metadata.get('supportsQuery'))
print("Max Record Count:", metadata.get('maxRecordCount'))

print("\nCapabilities:")
print("- Advanced Queries:", metadata.get('supportsAdvancedQueries'))
print("- Supports Statistics:", metadata.get('supportsStatistics'))

print("\n Fields:")
for field in metadata.get('fields', [])[:10]:
    print(f"  - {field['name']}: {field['type']}")

# Try different export formats
print("\n\n=== Testing different formats ===")
base_url = f"{service_url}/query"

formats_to_try = [
    ('geojson', 'geojson'),
    ('json', 'json'),
    ('pbf', 'pbf')
]

for fmt_name, fmt_code in formats_to_try:
    print(f"\nTrying format: {fmt_name}")
    params = {
        'where': 'OBJECTID=182',  # Single specific record
        'outFields': '*',
        'returnGeometry': 'true',
        'f': fmt_code,
    }

    try:
        r = requests.get(base_url, params=params, verify=False, timeout=30)
        if fmt_code == 'geojson':
            data = r.json()
            if 'features' in data:
                geom = data['features'][0].get('geometry') if len(data['features']) > 0 else None
                print(f"  Geometry: {geom}")
        elif fmt_code == 'json':
            data = r.json()
            if 'features' in data:
                feat = data['features'][0] if len(data['features']) > 0 else {}
                geom = feat.get('geometry')
                print(f"  Geometry: {geom}")
    except Exception as e:
        print(f"  Error: {e}")
