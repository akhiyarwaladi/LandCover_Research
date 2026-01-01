# Panduan Download Manual KLHK Data dengan Geometry

## Metode 1: Via KLHK WebGIS Portal (GRATIS)

### Step-by-Step:

1. **Akses Portal KLHK**
   - URL: https://geoportal.menlhk.go.id/
   - Atau: https://sigap.menlhk.go.id/

2. **Cari Layer Tutupan Lahan 2024**
   - Cari menu "Peta Interaktif" atau "Map Viewer"
   - Pilih layer: "PL2024_AR_250K" atau "Tutupan Lahan 2024"

3. **Export/Download**
   - Cari tombol "Export" atau "Download"
   - Pilih area: Provinsi Jambi
   - Format: Shapefile (.shp) atau GeoJSON
   - Filter: KODE_PROV = 15 (Jambi)

4. **Alternatif: Screenshot dan Digitasi Manual**
   - Jika tidak ada export, zoom ke area kecil
   - Export per tile/batch
   - Gabung manual menggunakan QGIS

---

## Metode 2: Kontak KLHK Langsung (GRATIS - OFFICIAL)

### Email Template:

```
Kepada: geoportal@menlhk.go.id
CC: info@menlhk.go.id
Subject: Permohonan Akses Data Spasial Tutupan Lahan 2024 untuk Penelitian

Yth. Tim Geoportal KLHK,

Saya [Nama], peneliti dari [Institusi], sedang melakukan penelitian
klasifikasi tutupan lahan Provinsi Jambi menggunakan citra satelit
Sentinel-2 dan machine learning.

Saya memerlukan data spasial Tutupan Lahan 2024 (PL2024_AR_250K)
untuk Provinsi Jambi dalam format Shapefile/GeoJSON dengan geometry
lengkap untuk keperluan:
1. Training data supervised classification
2. Validasi akurasi hasil klasifikasi
3. Publikasi penelitian ilmiah

Data atribut sudah berhasil saya download via API, namun geometry
tidak tersedia (null) untuk akses publik.

Mohon bantuan untuk:
- Akses enterprise/akademik ke layanan ArcGIS REST, atau
- File download langsung shapefile PL2024 Provinsi Jambi

Penelitian ini untuk kepentingan akademik non-komersial.

Terima kasih atas bantuan Bapak/Ibu.

Hormat saya,
[Nama]
[Email]
[Institusi]
```

**Response Time:** Biasanya 3-7 hari kerja
**Success Rate:** Tinggi untuk penelitian akademik

---

## Metode 3: Portal Satu Data Indonesia

### Portal Pemerintah:

1. **Tanahair Indonesia**
   - URL: https://tanahair.indonesia.go.id/sdi/
   - Dataset: "Tutupan Lahan"
   - Format: SHP, GeoJSON, GML
   - Status: Kadang tersedia, kadang tidak (tergantung update)

2. **Portal Data Indonesia**
   - URL: https://data.go.id/
   - Search: "tutupan lahan KLHK"
   - Filter: Shapefile format

3. **Portal Geospasial BIG**
   - URL: https://tanahair.indonesia.go.id/
   - Ina-Geoportal
   - Mungkin ada dataset KLHK

---

## Metode 4: Sumber Third-Party (BERBAYAR)

### LapakGIS
- **URL:** https://www.lapakgis.com/
- **Data:** KLHK Tutupan Lahan 2011-2024
- **Format:** Shapefile lengkap dengan geometry
- **Harga:** Berbayar (contact via WhatsApp)
- **Provinsi:** Bisa per provinsi (Jambi saja)

### TechnoGIS Indonesia
- **URL:** https://www.technogis.co.id/
- **Service:** Data spasial KLHK
- **Status:** Komersial

---

## Metode 5: GDAL/OGR Advanced (EXPERIMENTAL)

### Try WFS Service:

```python
from osgeo import ogr

# Try WFS endpoint
wfs_url = "WFS:https://geoportal.menlhk.go.id/geoserver/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetCapabilities"

ds = ogr.Open(wfs_url)
if ds:
    print("WFS accessible!")
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayer(i)
        print(f"Layer {i}: {layer.GetName()}")
```

### Try Direct Feature Access:

```python
import requests

# Try direct feature download (not query)
url = "https://geoportal.menlhk.go.id/server/rest/services/Time_Series/PL_2024/MapServer/0"

# Try export endpoint
export_url = f"{url}/export"
params = {
    'where': 'KODE_PROV=15',
    'outFields': '*',
    'f': 'shapefile'  # or 'geojson'
}

# This might work if export is enabled
```

---

## Metode 6: Use Older KLHK Data (2020-2023)

Data tahun sebelumnya mungkin masih accessible via API lama:

```bash
# Try 2023 data
python scripts/download_klhk.py --year 2023 --url <old_endpoint>

# Try 2022 data
python scripts/download_klhk.py --year 2022 --url <old_endpoint>
```

Lihat dokumentasi: https://lintasbumi.com/2022/03/28/mendownload-peta-penutupan-lahan-klhk-2020-langsung-dari-sumbernya/

---

## REKOMENDASI PRIORITAS:

### ⭐ Paling Mudah:
1. **Kontak KLHK via email** (gratis, official, success rate tinggi)
2. **Cek Portal Tanahair Indonesia** (mungkin ada)
3. **Manual download via WebGIS** (perlu waktu)

### 💰 Jika Urgent:
4. **LapakGIS** (berbayar tapi cepat, data terpercaya)

### 🔬 Eksperimental:
5. **GDAL WFS** (mungkin tidak work)
6. **Old endpoints** (mungkin sudah deprecated)

---

## Timeline Estimasi:

| Metode | Waktu | Biaya | Success Rate |
|--------|-------|-------|--------------|
| Email KLHK | 3-7 hari | Gratis | 80% |
| Portal Tanahair | 10 menit | Gratis | 30% (hit or miss) |
| Manual WebGIS | 2-4 jam | Gratis | 60% |
| LapakGIS | 1 hari | Rp. 50-200K | 95% |
| GDAL WFS | 30 menit | Gratis | 20% |

---

**NEXT ACTION:** Pilih metode mana yang mau dicoba dulu?
