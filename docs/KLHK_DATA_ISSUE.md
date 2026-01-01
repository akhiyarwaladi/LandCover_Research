# KLHK Data Download Issue - Geometry Access Restricted

## Tanggal: 2026-01-01

## Masalah

Server ArcGIS REST API KLHK **tidak mengirimkan geometry** meskipun parameter `returnGeometry=true` sudah diset dengan benar.

### Yang Sudah Dicoba:
1. ✅ GeoJSON format (`f=geojson`)
2. ✅ ESRI JSON format (`f=json`)
3. ✅ FeatureServer endpoint (error 500)
4. ✅ MapServer endpoint (geometry = null)
5. ✅ Spatial envelope filter
6. ✅ GDAL/OGR driver

### Hasil:
**Semua metode mengembalikan `geometry: null`**

## Root Cause

Berdasarkan riset web:
- **Server KLHK sekarang memerlukan enterprise login** untuk akses geometry
- Dulu (pre-2022) data bisa diakses publik secara penuh
- Sekarang hanya attributes yang bisa diakses tanpa autentikasi
- Geometry dibatasi untuk pengguna ter-autentikasi saja

**Sumber:**
- [Lintas Bumi Blog](https://lintasbumi.com/2022/03/28/mendownload-peta-penutupan-lahan-klhk-2020-langsung-dari-sumbernya/)
- Comments dari pembaca mengkonfirmasi server sekarang memerlukan enterprise login

## Data Yang Berhasil Di-download

File: `data/klhk/KLHK_PL2024_Jambi_Full.geojson` (3.9 MB)

### Isi:
- ✅ **28,100 polygons** untuk Provinsi Jambi
- ✅ **Attributes lengkap:**
  - OBJECTID
  - KODE_PROV (15 = Jambi)
  - PL2024_ID (kode kelas tutupan lahan)
  - PL2023_ID_R
- ❌ **Geometry: NULL** (tidak tersedia)

### Distribusi Kelas:
| Kode | Nama | Jumlah | % |
|------|------|--------|---|
| 2007 | Hutan Tanaman | 7,446 | 26.5% |
| 2002 | Hutan Lahan Kering Sekunder | 5,331 | 19.0% |
| 2010 | Perkebunan | 3,811 | 13.6% |
| 2014 | Tanah Terbuka | 3,751 | 13.3% |
| 20092 | Sawah | 2,786 | 9.9% |
| 2012 | Pemukiman | 2,064 | 7.3% |
| ... | ... | ... | ... |

## Solusi Alternatif

### Opsi 1: Kontak KLHK Langsung (RECOMMENDED)
- Email: geoportal@menlhk.go.id atau info@menlhk.go.id
- Request akses resmi atau download shapefile lengkap
- Jelaskan tujuan penelitian akademik

### Opsi 2: Gunakan Data Alternatif Untuk Ground Truth
Karena kita butuh ground truth untuk klasifikasi, bisa gunakan:

1. **ESA WorldCover 2021** (10m resolution)
   - URL: https://worldcover2021.esa.int/
   - Format: GeoTIFF
   - Free dan open access
   - 11 land cover classes

2. **Copernicus Global Land Cover** (100m)
   - URL: https://land.copernicus.eu/global/products/lc
   - Free access
   - Updated annually

3. **Google Dynamic World** (10m)
   - Via Google Earth Engine
   - Real-time land cover classification
   - 9 classes
   - Sudah punya contoh script di `gee_scripts/`

4. **MapBiomas Indonesia** (30m)
   - URL: https://mapbiomas.org/
   - Khusus Indonesia
   - Historical data 1985-sekarang

### Opsi 3: Gunakan Commercial/Third-Party Sources
- LapakGIS (https://www.lapakgis.com/) - mungkin berbayar
- Indonesia Geospasial Services

### Opsi 4: Manual Download Via WebGIS Interface
Coba akses https://geoportal.menlhk.go.id/ langsung:
1. Login (jika ada akun)
2. Cari layer PL2024_AR_250K
3. Lihat apakah ada tombol export/download
4. Download sebagai SHP/GeoJSON

### Opsi 5: Hybrid Approach (PRACTICAL SOLUTION)
Gunakan combination:
1. KLHK attributes data (yang sudah kita punya) untuk class labels
2. ESA WorldCover atau Dynamic World untuk spatial extent/geometry
3. Join berdasarkan spatial intersection

## Rekomendasi untuk Research

Mengingat kendala akses KLHK geometry:

**PILIHAN TERBAIK: Gunakan ESA WorldCover 2021 atau Dynamic World**

### Keuntungan:
- ✅ Open access, no restrictions
- ✅ Geometry lengkap
- ✅ Resolusi bagus (10m)
- ✅ Accepted secara internasional untuk penelitian
- ✅ Methodology transparent dan reproducible

### Update Research Narrative:
Alih-alih "menggunakan KLHK sebagai ground truth," update menjadi:

> "We use ESA WorldCover 2021 / Dynamic World as reference land cover data because:
> 1. It provides actual ground truth with validated accuracy
> 2. Open access ensures research reproducibility
> 3. 10m resolution matches Sentinel-2 imagery
> 4. While KLHK data would be ideal for Indonesia, geometry access is restricted to enterprise users"

## Next Steps

1. **Immediate:** Decide which alternative ground truth to use
2. **Short-term:** Update scripts to download ESA WorldCover or use Dynamic World
3. **Long-term:** Contact KLHK for official data access (untuk publikasi future)

## Scripts yang Perlu Di-update

Jika pakai alternatif data:
- `scripts/download_satellite.py` - sudah OK
- `scripts/land_cover_classification_klhk.py` - **perlu update** untuk use alternative ground truth
- `gee_scripts/g_earth_engine_improved.js` - sudah include Dynamic World!

## Catatan Penting

Data KLHK attributes yang sudah di-download **tetap valuable** untuk:
- Validasi class distribution
- Understanding KLHK classification scheme
- Potential future use jika geometry access granted
