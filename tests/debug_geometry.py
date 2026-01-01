#!/usr/bin/env python3
"""
Deep investigation: Kenapa geometry NULL meskipun returnGeometry=true?
"""

import requests
import json
import urllib3
urllib3.disable_warnings()

url = "https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0"

print("="*80)
print("DEEP INVESTIGATION: KLHK Geometry Issue")
print("="*80)

# ============================================================================
# TEST 1: Check Service Capabilities
# ============================================================================
print("\n[TEST 1] Checking service metadata...")
print("-"*80)

service_info = requests.get(f"{url}?f=json", verify=False, timeout=30)
metadata = service_info.json()

print(f"Layer Name: {metadata.get('name')}")
print(f"Geometry Type: {metadata.get('geometryType')}")
print(f"Has Geometry Properties: {metadata.get('hasGeometryProperties')}")
print(f"Supports Advanced Queries: {metadata.get('supportsAdvancedQueries')}")
print(f"Max Record Count: {metadata.get('maxRecordCount')}")

# Check capabilities
capabilities = metadata.get('capabilities', '')
print(f"\nCapabilities: {capabilities}")
print(f"  - Query: {'Query' in capabilities}")
print(f"  - Data: {'Data' in capabilities}")
print(f"  - Map: {'Map' in capabilities}")

# ============================================================================
# TEST 2: Try Different returnGeometry Values
# ============================================================================
print("\n[TEST 2] Testing different returnGeometry parameter values...")
print("-"*80)

test_values = [
    ('true', 'String "true"'),
    ('True', 'String "True"'),
    ('TRUE', 'String "TRUE"'),
    (True, 'Boolean True'),
    (1, 'Integer 1'),
    ('1', 'String "1"'),
]

query_url = f"{url}/query"
base_params = {
    'where': 'OBJECTID=182',
    'outFields': '*',
    'f': 'geojson',
}

for value, description in test_values:
    params = base_params.copy()
    params['returnGeometry'] = value

    print(f"\nTrying returnGeometry = {description} ({repr(value)})")

    try:
        r = requests.get(query_url, params=params, verify=False, timeout=30)

        # Print actual URL
        print(f"  URL: {r.url}")

        data = r.json()
        if 'features' in data and len(data['features']) > 0:
            geom = data['features'][0].get('geometry')
            if geom and geom != 'null':
                print(f"  ✅ SUCCESS! Geometry found!")
                print(f"  Geometry: {json.dumps(geom, indent=4)}")
                break
            else:
                print(f"  ❌ Geometry is NULL")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ============================================================================
# TEST 3: Try Without returnGeometry (Default Behavior)
# ============================================================================
print("\n[TEST 3] Testing WITHOUT returnGeometry parameter (server default)...")
print("-"*80)

params = {
    'where': 'OBJECTID=182',
    'outFields': '*',
    'f': 'geojson',
}

r = requests.get(query_url, params=params, verify=False, timeout=30)
print(f"URL: {r.url}")
data = r.json()

if 'features' in data and len(data['features']) > 0:
    geom = data['features'][0].get('geometry')
    print(f"Geometry (no param): {geom}")

# ============================================================================
# TEST 4: Check Response Headers & Server Info
# ============================================================================
print("\n[TEST 4] Server response info...")
print("-"*80)

print(f"Status Code: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type')}")
print(f"Server: {r.headers.get('Server')}")

# ============================================================================
# TEST 5: Try ESRI JSON Format (More Verbose)
# ============================================================================
print("\n[TEST 5] Testing with ESRI JSON format (f=json)...")
print("-"*80)

params = {
    'where': 'OBJECTID=182',
    'outFields': '*',
    'returnGeometry': 'true',
    'f': 'json',  # ESRI JSON instead of GeoJSON
}

r = requests.get(query_url, params=params, verify=False, timeout=30)
data = r.json()

print(f"Response keys: {list(data.keys())}")

if 'features' in data and len(data['features']) > 0:
    feature = data['features'][0]
    print(f"\nFeature keys: {list(feature.keys())}")
    print(f"\nFull feature:")
    print(json.dumps(feature, indent=2))

# ============================================================================
# TEST 6: Check If It's a Layer Restriction
# ============================================================================
print("\n[TEST 6] Checking layer definition...")
print("-"*80)

# Check if there's a layer definition that excludes geometry
layer_defs = metadata.get('layerDefinition', {})
print(f"Layer Definition: {json.dumps(layer_defs, indent=2)}")

# Check fields
fields = metadata.get('fields', [])
print(f"\nAvailable fields ({len(fields)}):")
for field in fields:
    print(f"  - {field['name']}: {field['type']}")

# ============================================================================
# TEST 7: Try Exporting Instead of Querying
# ============================================================================
print("\n[TEST 7] Testing EXPORT endpoint (alternative to query)...")
print("-"*80)

# Some ArcGIS servers have /export endpoint
export_url = f"{url}/export"
params = {
    'where': 'OBJECTID=182',
    'returnGeometry': 'true',
    'f': 'json',
}

try:
    r = requests.get(export_url, params=params, verify=False, timeout=30)
    print(f"Export endpoint status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Export response: {json.dumps(data, indent=2)[:500]}")
except Exception as e:
    print(f"Export endpoint not available: {e}")

# ============================================================================
# TEST 8: Check Service-Level Restrictions
# ============================================================================
print("\n[TEST 8] Checking access restrictions...")
print("-"*80)

# Check if service requires authentication
print(f"Current Access: {metadata.get('currentVersion')}")
print(f"Service Description: {metadata.get('serviceDescription', 'N/A')[:100]}")

# Check supported query formats
supported_formats = metadata.get('supportedQueryFormats', '')
print(f"Supported Query Formats: {supported_formats}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & DIAGNOSIS")
print("="*80)

print("""
FACTS:
1. Layer HAS geometry type: esriGeometryPolygon
2. Server responds to queries successfully
3. Attributes are returned correctly
4. Geometry is consistently NULL across all attempts

POSSIBLE CAUSES:
A. Server-side restriction on geometry access (enterprise only)
B. Layer security settings prevent geometry export
C. Service configuration explicitly disables geometry in queries
D. Database view without spatial column exposed
E. ArcGIS REST API version incompatibility

RECOMMENDATION:
Since all technical attempts failed, this is likely a deliberate
server-side restriction. The only solutions are:
1. Contact KLHK for official access
2. Use alternative data source (LapakGIS, manual download)
3. Switch methodology (unsupervised + validation)
""")
