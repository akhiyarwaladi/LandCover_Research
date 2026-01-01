import requests
import json
import urllib3
urllib3.disable_warnings()

# Check if there's a WFS or download service
print("Checking for alternative download methods...")
print("="*60)

# 1. Check service capabilities
service_url = 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer'
r = requests.get(f"{service_url}?f=json", verify=False)
data = r.json()

print("\nService Capabilities:")
print(f"- supportsDynamicLayers: {data.get('supportsDynamicLayers')}")
print(f"- supportsQueryDataElements: {data.get('supportsQueryDataElements')}")
print(f"- exportTilesAllowed: {data.get('exportTilesAllowed')}")

# 2. Try to access layer0 with different approach - using bounding box
print("\n\nTrying with spatial extent filter...")
layer_url = f"{service_url}/0"
r = requests.get(f"{layer_url}?f=json", verify=False)
layer_info = r.json()

extent = layer_info.get('extent')
print(f"Layer extent: {extent}")

# Try querying with explicit geometry/envelope
query_url = f"{layer_url}/query"
params = {
    'where': 'KODE_PROV=15',
    'geometry': json.dumps({
        'xmin': extent['xmin'],
        'ymin': extent['ymin'],
        'xmax': extent['xmax'],
        'ymax': extent['ymax'],
        'spatialReference': {'wkid': extent['spatialReference']['wkid']}
    }),
    'geometryType': 'esriGeometryEnvelope',
    'spatialRel': 'esriSpatialRelIntersects',
    'outFields': '*',
    'returnGeometry': 'true',
    'f': 'geojson',
    'resultRecordCount': 1
}

print("\nTrying with envelope geometry...")
r = requests.get(query_url, params=params, verify=False, timeout=30)
if r.status_code == 200:
    result = r.json()
    if 'features' in result and len(result['features']) > 0:
        geom = result['features'][0].get('geometry')
        print(f"Geometry: {geom}")


# 3. Check if OGC WFS is available
print("\n\nChecking for WFS service...")
wfs_url = 'https://geoportal.menlhk.go.id/geoserver/wfs'
params = {
    'service': 'WFS',
    'version': '2.0.0',
    'request': 'GetCapabilities'
}
try:
    r = requests.get(wfs_url, params=params, verify=False, timeout=10)
    if r.status_code == 200 and 'WFS_Capabilities' in r.text:
        print("✅ WFS service found!")
        print(f"Status: {r.status_code}")
    else:
        print("❌ No WFS service")
except:
    print("❌ No WFS service available")
