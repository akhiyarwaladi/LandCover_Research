/**
 * Download Multi-Temporal Sentinel-2 Annual Composites for Jambi Province
 *
 * Creates 7 annual dry-season (June-October) median composites (2018-2024)
 * using Cloud Score+ masking for cloud-free imagery.
 *
 * Bands: B2, B3, B4, B5, B6, B7, B8A, B11, B12 (10 bands at 20m)
 * Resolution: 20 meters
 * CRS: EPSG:4326
 *
 * Usage: Paste into Google Earth Engine Code Editor and run.
 *        Exports will appear in the Tasks tab.
 *
 * Output: 7 GeoTIFF files on Google Drive:
 *   S2_jambi_2018_20m_AllBands.tif
 *   S2_jambi_2019_20m_AllBands.tif
 *   ...
 *   S2_jambi_2024_20m_AllBands.tif
 */

// ============================================================
// 1. Define Study Area: Jambi Province, Sumatra, Indonesia
// ============================================================

var jambi = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM1_NAME', 'Jambi'));

var jambiBounds = jambi.geometry();

// Visualization check
Map.centerObject(jambiBounds, 8);
Map.addLayer(jambiBounds, {color: 'red'}, 'Jambi Province');

// ============================================================
// 2. Parameters
// ============================================================

var YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024];
var DRY_SEASON_START = '-06-01';
var DRY_SEASON_END = '-11-01';
var CLOUD_THRESHOLD = 0.60;
var SCALE = 20;  // 20m resolution
var CRS = 'EPSG:4326';

// Bands to export (10 bands at 20m)
var BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12'];

// ============================================================
// 3. Cloud Masking Function using Cloud Score+
// ============================================================

function maskClouds(image) {
  // Use Cloud Score+ for cloud masking
  var csPlus = ee.Image('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED/' +
                         image.get('system:index'));
  var cloudMask = csPlus.select('cs').gte(CLOUD_THRESHOLD);
  return image.updateMask(cloudMask);
}

// Alternative: SCL-based masking (fallback for older imagery)
function maskCloudsSCL(image) {
  var scl = image.select('SCL');
  // Keep: vegetation (4), bare soil (5), water (6), snow (11)
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(11));
  return image.updateMask(mask);
}

// ============================================================
// 4. Process Each Year
// ============================================================

YEARS.forEach(function(year) {
  var startDate = year + DRY_SEASON_START;
  var endDate = year + DRY_SEASON_END;

  // Load Sentinel-2 SR Harmonized
  var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(jambiBounds)
      .filterDate(startDate, endDate)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50));

  // Print collection size for verification
  print('Year ' + year + ' - Image count:', s2.size());

  // Apply cloud masking
  var s2Masked;
  if (year >= 2020) {
    // Cloud Score+ available from ~2020
    s2Masked = s2.map(function(image) {
      try {
        return maskClouds(image);
      } catch(e) {
        return maskCloudsSCL(image);
      }
    });
  } else {
    // Use SCL for older imagery
    s2Masked = s2.map(maskCloudsSCL);
  }

  // Create median composite
  var composite = s2Masked
      .select(BANDS)
      .median()
      .clip(jambiBounds)
      .toFloat();

  // Add to map for visual check
  var visParams = {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000};
  Map.addLayer(composite, visParams, 'S2 ' + year, false);

  // Export to Google Drive
  var fileName = 'S2_jambi_' + year + '_20m_AllBands';

  Export.image.toDrive({
    image: composite,
    description: fileName,
    folder: 'GEE_Deforestation_Jambi',
    fileNamePrefix: fileName,
    region: jambiBounds,
    scale: SCALE,
    crs: CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });

  print('Export task created: ' + fileName);
});

// ============================================================
// 5. Summary
// ============================================================

print('=== Multi-Temporal Sentinel-2 Download ===');
print('Study area: Jambi Province, Sumatra');
print('Years: 2018-2024 (7 annual composites)');
print('Season: June-October (dry season)');
print('Bands: ' + BANDS.join(', '));
print('Resolution: ' + SCALE + 'm');
print('CRS: ' + CRS);
print('');
print('Click RUN on each export task in the Tasks tab.');
print('Files will be saved to Google Drive: GEE_Deforestation_Jambi/');
