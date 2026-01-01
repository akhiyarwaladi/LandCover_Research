import requests
import json
import urllib3
urllib3.disable_warnings()

# Try FeatureServer instead of MapServer
urls_to_try = [
    'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query',
    'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/FeatureServer/0/query',
]

for url in urls_to_try:
    print(f"\n{'='*60}")
    print(f"Testing: {url}")
    print('='*60)

    params = {
        'where': 'OBJECTID=182',
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'geojson',
        'geometryPrecision': 6
    }

    try:
        r = requests.get(url, params=params, verify=False, timeout=30)
        print(f"Status Code: {r.status_code}")

        if r.status_code == 200:
            data = r.json()
            if 'features' in data and len(data['features']) > 0:
                feat = data['features'][0]
                geom = feat.get('geometry')
                print(f"Geometry type: {type(geom)}")
                print(f"Geometry value: {geom}")

                if geom and geom != 'null':
                    print("\n✅ SUCCESS! Geometry found!")
                    print(f"Geometry keys: {list(geom.keys()) if isinstance(geom, dict) else 'Not a dict'}")
                else:
                    print("\n❌ Geometry is None or null")
        else:
            print(f"Error response: {r.text[:500]}")
    except Exception as e:
        print(f"Exception: {e}")
