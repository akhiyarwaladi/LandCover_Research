// ============================================================================
// IMPROVED LAND COVER ANALYSIS SCRIPT FOR JAMBI PROVINCE
// Using Sentinel-2 + Dynamic World + Cloud Score+
// Updated: December 2024
// ============================================================================

// =========================
// CONFIGURATION PARAMETERS
// =========================
var CONFIG = {
  // Temporal settings
  startDate: '2024-01-01',
  endDate: '2024-12-31',

  // Cloud filtering
  maxCloudPercent: 20,        // Pre-filter cloud percentage
  cloudScoreThreshold: 0.60,  // Cloud Score+ threshold (0.5-0.65 recommended)

  // Export settings
  exportScale: 10,            // Resolution in meters
  exportFolder: 'GEE_Exports',
  exportCRS: 'EPSG:4326',

  // Region name for export filenames
  regionName: 'jambi',
  yearLabel: '2024'
};

// =========================
// DEFINE AREA OF INTEREST
// =========================
// IMPORTANT: Run the verification script first to confirm exact field values!
// See: verification_boundaries.js

// ---- CONFIGURATION ----
var USE_BOUNDARY_SOURCE = 'GAUL';  // Options: 'GAUL', 'GEOBOUNDARIES', 'CUSTOM', 'BBOX'
var PROVINCE_NAME = 'Jambi';       // Adjust if needed after verification

// ---- BOUNDARY SELECTION ----
var jambi;

if (USE_BOUNDARY_SOURCE === 'GAUL') {
  // Option 1: FAO GAUL 2015 (RECOMMENDED - Well documented, reliable)
  // License: Non-commercial only
  // Docs: https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level1
  var gaulCollection = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))
    .filter(ee.Filter.eq('ADM1_NAME', PROVINCE_NAME));

  jambi = gaulCollection.geometry();
  print('Using FAO GAUL boundary for:', PROVINCE_NAME);
  print('Features found:', gaulCollection.size());

} else if (USE_BOUNDARY_SOURCE === 'GEOBOUNDARIES') {
  // Option 2: geoBoundaries v6.0 (Updated 2023, CC BY 4.0)
  // License: Open for commercial use
  // Docs: https://developers.google.com/earth-engine/datasets/catalog/WM_geoLab_geoBoundaries_600_ADM1
  // NOTE: Verify shapeName value first using verification script!
  var gbCollection = ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM1')
    .filter(ee.Filter.eq('shapeGroup', 'IDN'))
    .filter(ee.Filter.eq('shapeName', PROVINCE_NAME));

  jambi = gbCollection.geometry();
  print('Using geoBoundaries for:', PROVINCE_NAME);
  print('Features found:', gbCollection.size());

} else if (USE_BOUNDARY_SOURCE === 'CUSTOM') {
  // Option 3: Custom shapefile from BIG (Most accurate for Indonesia)
  // Upload your shapefile to GEE Assets first
  // Download from: https://tanahair.indonesia.go.id/
  jambi = ee.FeatureCollection('users/YOUR_USERNAME/jambi_boundary').geometry();
  print('Using custom boundary');

} else {
  // Option 4: Bounding Box (Fallback - for testing only)
  // WARNING: Includes area outside actual province boundary!
  jambi = ee.Geometry.Rectangle([102.5, -2.6, 104.6, -0.8]);
  print('WARNING: Using bounding box - not recommended for final analysis!');
}

// Validate that geometry is not empty
var areaCheck = jambi.area().divide(1e6);  // km²
print('AOI Area (km²):', areaCheck);

// =========================
// LOAD DATASETS
// =========================
// Load Sentinel-2 Surface Reflectance (HARMONIZED - recommended since 2024)
var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(CONFIG.startDate, CONFIG.endDate)
  .filterBounds(jambi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.maxCloudPercent));

// Load Cloud Score+ dataset for advanced cloud masking
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
  .filterDate(CONFIG.startDate, CONFIG.endDate)
  .filterBounds(jambi);

// Load Dynamic World
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterDate(CONFIG.startDate, CONFIG.endDate)
  .filterBounds(jambi);

// =========================
// CLOUD MASKING FUNCTION
// Using Cloud Score+ (Best practice 2024)
// =========================
function maskCloudsWithCSPlus(image) {
  // Get the corresponding Cloud Score+ image
  var csImage = csPlus
    .filter(ee.Filter.eq('system:index', image.get('system:index')))
    .first();

  // Use cs_cdf band (cumulative distribution function - more robust)
  var cs = csImage.select('cs_cdf');

  // Apply threshold mask
  var clearMask = cs.gte(CONFIG.cloudScoreThreshold);

  return image
    .updateMask(clearMask)
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    .divide(10000)
    .copyProperties(image, ['system:time_start']);
}

// Alternative: SCL-based cloud masking (fallback)
function maskCloudsWithSCL(image) {
  var scl = image.select('SCL');

  // Keep only: vegetation (4), bare soil (5), water (6), unclassified (7)
  var clearMask = scl.gte(4).and(scl.lte(7));

  // Also use cloud probability if available
  var cloudProb = image.select('MSK_CLDPRB');
  var probMask = cloudProb.lt(40);

  var finalMask = clearMask.and(probMask);

  return image
    .updateMask(finalMask)
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    .divide(10000)
    .copyProperties(image, ['system:time_start']);
}

// =========================
// APPLY CLOUD MASKING
// =========================
// Try Cloud Score+ first, fallback to SCL if not available
var s2Masked = s2Sr.map(function(image) {
  var csImage = csPlus
    .filter(ee.Filter.eq('system:index', image.get('system:index')))
    .first();

  return ee.Algorithms.If(
    csImage,
    maskCloudsWithCSPlus(image),
    maskCloudsWithSCL(image)
  );
});

// Convert to ImageCollection
s2Masked = ee.ImageCollection(s2Masked.filter(ee.Filter.notNull(['system:time_start'])));

// =========================
// CREATE COMPOSITES
// =========================
var s2Composite = s2Masked.median().clip(jambi);

// Dynamic World composites
var dwLabel = dw.select('label').mode().clip(jambi);
var dwProbability = dw.select([
  'water', 'trees', 'grass', 'flooded_vegetation',
  'crops', 'shrub_and_scrub', 'built', 'bare', 'snow_and_ice'
]).mean().clip(jambi);

// Apply smoothing to reduce salt-and-pepper noise
var dwSmoothed = dwLabel.focal_mode({
  kernel: ee.Kernel.circle({radius: 1}),
  iterations: 2
});

// =========================
// CALCULATE SPECTRAL INDICES
// =========================
var indices = ee.Image.cat([
  // NDVI - Normalized Difference Vegetation Index
  s2Composite.normalizedDifference(['B8', 'B4']).rename('NDVI'),

  // EVI - Enhanced Vegetation Index (better for dense vegetation)
  s2Composite.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': s2Composite.select('B8'),
      'RED': s2Composite.select('B4'),
      'BLUE': s2Composite.select('B2')
    }).rename('EVI'),

  // NDWI - Normalized Difference Water Index (water bodies)
  s2Composite.normalizedDifference(['B3', 'B8']).rename('NDWI'),

  // NDMI - Normalized Difference Moisture Index (vegetation moisture)
  s2Composite.normalizedDifference(['B8', 'B11']).rename('NDMI'),

  // MNDWI - Modified NDWI (better water detection)
  s2Composite.normalizedDifference(['B3', 'B11']).rename('MNDWI'),

  // NDBI - Normalized Difference Built-up Index
  s2Composite.normalizedDifference(['B11', 'B8']).rename('NDBI'),

  // SAVI - Soil Adjusted Vegetation Index
  s2Composite.expression(
    '((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
      'NIR': s2Composite.select('B8'),
      'RED': s2Composite.select('B4')
    }).rename('SAVI'),

  // NBR - Normalized Burn Ratio (for fire/burn detection)
  s2Composite.normalizedDifference(['B8', 'B12']).rename('NBR')
]);

// =========================
// QUALITY METRICS
// =========================
var imageCount = s2Masked.size();
print('Number of cloud-free images used:', imageCount);

// Calculate pixel-wise observation count
var obsCount = s2Masked.select('B4').count().rename('observation_count');

// =========================
// VISUALIZATION PARAMETERS
// =========================
var rgbVis = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.3,
  gamma: 1.2
};

var fcVis = {
  bands: ['B8', 'B4', 'B3'],  // False color (vegetation in red)
  min: 0,
  max: 0.4,
  gamma: 1.1
};

var dwVis = {
  min: 0,
  max: 8,
  palette: [
    '#419BDF',  // 0: water
    '#397D49',  // 1: trees
    '#88B053',  // 2: grass
    '#7A87C6',  // 3: flooded_vegetation
    '#E49635',  // 4: crops
    '#DFC35A',  // 5: shrub_and_scrub
    '#C4281B',  // 6: built
    '#A59B8F',  // 7: bare
    '#B39FE1'   // 8: snow_and_ice
  ]
};

var ndviVis = {
  min: -0.2,
  max: 0.8,
  palette: ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
};

// =========================
// ADD MAP LAYERS
// =========================
Map.centerObject(jambi, 9);
Map.addLayer(s2Composite, rgbVis, 'Sentinel-2 RGB', true);
Map.addLayer(s2Composite, fcVis, 'Sentinel-2 False Color', false);
Map.addLayer(dwSmoothed, dwVis, 'Dynamic World (Smoothed)', true);
Map.addLayer(dwLabel, dwVis, 'Dynamic World (Raw)', false);
Map.addLayer(indices.select('NDVI'), ndviVis, 'NDVI', false);
Map.addLayer(indices.select('EVI'), ndviVis, 'EVI', false);
Map.addLayer(indices.select('NDWI'), {min: -0.5, max: 0.5, palette: ['brown', 'white', 'blue']}, 'NDWI', false);
Map.addLayer(indices.select('NDBI'), {min: -0.3, max: 0.3, palette: ['green', 'white', 'red']}, 'NDBI', false);
Map.addLayer(obsCount, {min: 1, max: 50, palette: ['red', 'yellow', 'green']}, 'Observation Count', false);

// Add legend
print('Dynamic World Classes:');
print('0: Water, 1: Trees, 2: Grass, 3: Flooded Vegetation');
print('4: Crops, 5: Shrub/Scrub, 6: Built, 7: Bare, 8: Snow/Ice');

// =========================
// EXPORT FUNCTIONS
// =========================
var exportParams = {
  crs: CONFIG.exportCRS,
  scale: CONFIG.exportScale,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
};

// Export Sentinel-2 All Bands (10m native bands + 20m resampled)
Export.image.toDrive({
  image: s2Composite.select(['B2', 'B3', 'B4', 'B8']).toFloat(),
  description: 'S2_' + CONFIG.regionName + '_' + CONFIG.yearLabel + '_10m_RGBNIR',
  folder: CONFIG.exportFolder,
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export Sentinel-2 20m bands (Red Edge + SWIR)
Export.image.toDrive({
  image: s2Composite.select(['B5', 'B6', 'B7', 'B8A', 'B11', 'B12']).toFloat(),
  description: 'S2_' + CONFIG.regionName + '_' + CONFIG.yearLabel + '_20m_RedEdgeSWIR',
  folder: CONFIG.exportFolder,
  scale: 20,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export Dynamic World Classification
Export.image.toDrive({
  image: dwSmoothed.toByte(),
  description: 'DW_' + CONFIG.regionName + '_' + CONFIG.yearLabel + '_classification',
  folder: CONFIG.exportFolder,
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export Dynamic World Probabilities
Export.image.toDrive({
  image: dwProbability.toFloat(),
  description: 'DW_' + CONFIG.regionName + '_' + CONFIG.yearLabel + '_probabilities',
  folder: CONFIG.exportFolder,
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export All Spectral Indices
Export.image.toDrive({
  image: indices.toFloat(),
  description: 'Indices_' + CONFIG.regionName + '_' + CONFIG.yearLabel + '_all',
  folder: CONFIG.exportFolder,
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export Observation Count (QC layer)
Export.image.toDrive({
  image: obsCount.toInt16(),
  description: 'QC_' + CONFIG.regionName + '_' + CONFIG.yearLabel + '_obsCount',
  folder: CONFIG.exportFolder,
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

print('=== EXPORT TASKS READY ===');
print('Go to Tasks tab to run exports');
print('Estimated files: 6 GeoTIFFs');
