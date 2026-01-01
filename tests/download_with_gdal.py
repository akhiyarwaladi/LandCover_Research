#!/usr/bin/env python3
"""
Download KLHK data using GDAL OGR Driver for ArcGIS REST
This bypasses some API limitations
"""
from osgeo import ogr, gdal
import os

# Enable GDAL exceptions
gdal.UseExceptions()

# ArcGIS REST endpoint
service_url = "https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0"

# ESRI JSON format connection string
# Format: ESRIJSON:http://...
connection_string = f"ESRIJSON:{service_url}/query?where=KODE_PROV%3D15&outFields=*&f=json"

print("Attempting to access ArcGIS REST service using GDAL OGR...")
print(f"URL: {service_url}")
print("="*80)

try:
    # Open the data source
    print("\n[1/4] Opening data source...")
    ds = ogr.Open(connection_string)

    if ds is None:
        print("❌ Failed to open data source")
        print("\nTrying alternative connection method...")

        # Try with OGRGeoJSON driver
        import requests
        import urllib3
        urllib3.disable_warnings()

        # Download a small batch first to test
        query_url = f"{service_url}/query"
        params = {
            'where': 'KODE_PROV=15',
            'outFields': '*',
            'returnGeometry': 'true',
            'f': 'json',  # Try ESRI JSON instead of GeoJSON
            'resultRecordCount': 5
        }

        print("Fetching test data in ESRI JSON format...")
        r = requests.get(query_url, params=params, verify=False, timeout=30)
        esri_json = r.json()

        print(f"Response keys: {list(esri_json.keys())}")

        if 'features' in esri_json:
            features = esri_json['features']
            print(f"Features count: {len(features)}")

            if len(features) > 0:
                first_feat = features[0]
                print(f"\nFirst feature keys: {list(first_feat.keys())}")
                print(f"Geometry: {first_feat.get('geometry')}")

                # Check if geometry exists in ESRI JSON format
                if first_feat.get('geometry'):
                    print("\n✅ GEOMETRY FOUND in ESRI JSON format!")
                    print(f"Geometry type: {first_feat['geometry'].get('type') if isinstance(first_feat.get('geometry'), dict) else type(first_feat.get('geometry'))}")

                    # Now download all data
                    print("\n\n[2/4] Geometry confirmed! Downloading all data...")
                    all_features = []
                    offset = 0
                    step = 1000

                    # Get total count
                    count_params = {
                        'where': 'KODE_PROV=15',
                        'returnCountOnly': 'true',
                        'f': 'json'
                    }
                    r_count = requests.get(query_url, params=count_params, verify=False)
                    total = r_count.json().get('count', 0)
                    print(f"Total features to download: {total}")

                    while True:
                        params['resultOffset'] = offset
                        params['resultRecordCount'] = step
                        params.pop('resultRecordCount', None)
                        params['resultRecordCount'] = step

                        print(f"Downloading {offset}-{min(offset+step, total)} / {total}")

                        r = requests.get(query_url, params=params, verify=False, timeout=60)
                        data = r.json()
                        features = data.get('features', [])

                        if not features:
                            break

                        all_features.extend(features)

                        if len(features) < step:
                            break

                        offset += step

                    print(f"\n[3/4] Downloaded {len(all_features)} features")

                    # Convert ESRI JSON to GeoJSON
                    print("\n[4/4] Converting to GeoJSON...")
                    import json

                    # Convert ESRI JSON features to GeoJSON
                    geojson_features = []
                    for esri_feat in all_features:
                        geom = esri_feat.get('geometry')
                        props = esri_feat.get('attributes', {})

                        # Convert ESRI geometry to GeoJSON geometry
                        geojson_geom = None
                        if geom:
                            if 'rings' in geom:  # Polygon
                                geojson_geom = {
                                    'type': 'Polygon',
                                    'coordinates': geom['rings']
                                }
                            elif 'paths' in geom:  # LineString
                                geojson_geom = {
                                    'type': 'LineString',
                                    'coordinates': geom['paths'][0] if geom['paths'] else []
                                }
                            elif 'x' in geom and 'y' in geom:  # Point
                                geojson_geom = {
                                    'type': 'Point',
                                    'coordinates': [geom['x'], geom['y']]
                                }

                        geojson_feat = {
                            'type': 'Feature',
                            'geometry': geojson_geom,
                            'properties': props
                        }
                        geojson_features.append(geojson_feat)

                    geojson = {
                        'type': 'FeatureCollection',
                        'name': 'KLHK_PL2024_Jambi',
                        'crs': {
                            'type': 'name',
                            'properties': {'name': 'urn:ogc:def:crs:EPSG::4326'}
                        },
                        'features': geojson_features
                    }

                    # Save to file
                    output_file = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
                    with open(output_file, 'w') as f:
                        json.dump(geojson, f)

                    file_size = os.path.getsize(output_file) / (1024 * 1024)
                    print(f"\n✅ SUCCESS!")
                    print(f"File saved: {output_file}")
                    print(f"File size: {file_size:.2f} MB")
                    print(f"Features: {len(geojson_features)}")
                else:
                    print("\n❌ No geometry in ESRI JSON either")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
