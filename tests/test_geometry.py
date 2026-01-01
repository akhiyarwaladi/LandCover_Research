import requests
import json
import urllib3
urllib3.disable_warnings()

url = 'https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0/query'
params = {
    'where': 'KODE_PROV=15',
    'outFields': '*',
    'returnGeometry': 'true',
    'f': 'geojson',
    'resultRecordCount': 1
}

print("Testing geometry download...")
r = requests.get(url, params=params, verify=False, timeout=30)
data = r.json()
print('Response keys:', list(data.keys()))
if 'features' in data and len(data['features']) > 0:
    feat = data['features'][0]
    print('\nFirst feature keys:', list(feat.keys()))
    print('\nGeometry value:', feat.get('geometry'))
    print('\nProperties:', feat.get('properties'))
else:
    print('No features found!')
