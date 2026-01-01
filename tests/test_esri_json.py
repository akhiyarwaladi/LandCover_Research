import requests
import json
import urllib3
urllib3.disable_warnings()

url = 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query'

# Try ESRI JSON format (f=json) instead of GeoJSON (f=geojson)
params = {
    'where': 'KODE_PROV=15',
    'outFields': '*',
    'returnGeometry': 'true',
    'f': 'json',  # ESRI JSON format
    'resultRecordCount': 2
}

print("Testing ESRI JSON format...")
print("="*60)
r = requests.get(url, params=params, verify=False, timeout=30)
data = r.json()

print(f"Response keys: {list(data.keys())}")

if 'features' in data:
    features = data['features']
    print(f"\nTotal features in response: {len(features)}")

    if features:
        feat = features[0]
        print(f"\nFirst feature keys: {list(feat.keys())}")
        print(f"\nGeometry value:")
        print(json.dumps(feat.get('geometry'), indent=2))

        if feat.get('geometry'):
            print("\n✅ GEOMETRY EXISTS in ESRI JSON format!")
            geom = feat['geometry']
            if 'rings' in geom:
                print(f"Polygon with {len(geom['rings'])} rings")
                print(f"First ring has {len(geom['rings'][0])} points")
        else:
            print("\n❌ Geometry is still null in ESRI JSON")
