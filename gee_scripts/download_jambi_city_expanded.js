// ============================================================================
// Download Sentinel-2 for Jambi City (EXPANDED to Legacy BBox Size)
// ============================================================================
//
// Option 3: Envelope of City + Proportional Expansion
// - Starts from Jambi City administrative center
// - Expands to match legacy bounding box size
// - Clips to province administrative boundaries
// - Maintains natural/curved boundaries (NOT rectangle)
//
// Author: Claude Sonnet 4.5
// Date: 2026-01-03
// ============================================================================

// ============================================================================
// CONFIGURATION
// ============================================================================

// Legacy bounding box size (for reference)
var LEGACY_BBOX = ee.Geometry.Rectangle([103.4486, -1.8337, 103.7566, -1.4089]);
var LEGACY_SIZE_DEG = {
  width: 103.7566 - 103.4486,   // ~0.308°
  height: -1.4089 - (-1.8337)    // ~0.425°
};

print('Legacy BBox Size:', LEGACY_SIZE_DEG);

// Expansion factor (adjust this to match desired area)
var EXPANSION_FACTOR = 1.8;  // Make it 80% larger than city center

// Export parameters
var EXPORT_SCALE = 20;  // 20m resolution (for all 10 bands)
var EXPORT_FOLDER = 'GEE_Exports';
var EXPORT_CRS = 'EPSG:4326';

// ============================================================================
// STEP 1: Get Jambi Province Administrative Boundary
// ============================================================================

var jambiProvince = ee.FeatureCollection("FAO/GAUL/2015/level1")
    .filter(ee.Filter.eq('ADM1_NAME', 'Jambi'));

print('Jambi Province loaded');
Map.addLayer(jambiProvince, {color: 'lightgreen'}, 'Jambi Province', false);

// ============================================================================
// STEP 2: Get Jambi City Center Point
// ============================================================================

// Jambi City approximate center (you can adjust this)
var jambiCityCenter = ee.Geometry.Point([103.6167, -1.6000]);

// Or use administrative boundary if available
// var jambiCity = ee.FeatureCollection("FAO/GAUL/2015/level2")
//     .filter(ee.Filter.eq('ADM1_NAME', 'Jambi'))
//     .filter(ee.Filter.eq('ADM2_NAME', 'Jambi'));  // City name
// var jambiCityCenter = jambiCity.geometry().centroid();

Map.addLayer(jambiCityCenter, {color: 'red'}, 'Jambi City Center');

// ============================================================================
// STEP 3: Create Expanded Region (Option 3)
// ============================================================================

// Calculate buffer distance to match legacy bbox size
// Legacy bbox is ~0.308° × 0.425°
// At equator, 1° ≈ 111 km
// Target size: ~34 km × 47 km
var targetSizeKm = Math.sqrt(34 * 47);  // geometric mean ≈ 40 km
var bufferDistanceKm = (targetSizeKm / 2) * EXPANSION_FACTOR;
var bufferDistanceMeters = bufferDistanceKm * 1000;

print('Buffer distance:', bufferDistanceKm, 'km');

// Create circular buffer around city center
var cityExpanded = jambiCityCenter.buffer(bufferDistanceMeters);

// Clip to province boundaries (maintain administrative shape)
var jambiRegion = cityExpanded.intersection(jambiProvince.geometry(), 1);

// Visualize
Map.centerObject(jambiRegion, 10);
Map.addLayer(LEGACY_BBOX, {color: 'red'}, 'Legacy BBox (Reference)', false);
Map.addLayer(cityExpanded, {color: 'orange'}, 'Circular Expansion', false);
Map.addLayer(jambiRegion, {color: 'blue'}, 'FINAL Region (Clipped to Province)');

// Get bounds for comparison
var regionBounds = jambiRegion.bounds();
print('Final Region Bounds:', regionBounds.coordinates());

// ============================================================================
// STEP 4: Cloud Masking Function
// ============================================================================

function maskS2clouds(image) {
  var scl = image.select('SCL');

  // Cloud, shadow, snow mask
  var cloudMask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9))
      .and(scl.neq(1).and(scl.neq(2)));

  // Cloud probability
  var cloudProb = image.select('MSK_CLDPRB');
  var cloudProbMask = cloudProb.lt(40);

  var finalMask = cloudMask.and(cloudProbMask);

  return image.updateMask(finalMask)
      .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
      .divide(10000);
}

// ============================================================================
// STEP 5: Load and Process Sentinel-2 Data
// ============================================================================

var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2024-01-01', '2024-12-31')
    .filterBounds(jambiRegion)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2clouds);

var s2Composite = s2.median();

// ============================================================================
// STEP 6: Visualization
// ============================================================================

var rgbVis = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.3,
  gamma: 1.2
};

Map.addLayer(s2Composite.clip(jambiRegion), rgbVis, 'Sentinel-2 RGB');

// NDVI for quality check
var ndvi = s2Composite.normalizedDifference(['B8', 'B4']);
Map.addLayer(ndvi.clip(jambiRegion),
  {min: -1, max: 1, palette: ['red', 'yellow', 'green']},
  'NDVI', false);

// ============================================================================
// STEP 7: Export
// ============================================================================

// Export Sentinel-2 (all 10 bands at 20m)
Export.image.toDrive({
  image: s2Composite.select([
    'B2', 'B3', 'B4',     // RGB
    'B5', 'B6', 'B7',     // Red Edge
    'B8', 'B8A',          // NIR
    'B11', 'B12'          // SWIR
  ]).clip(jambiRegion).reproject({
    crs: EXPORT_CRS,
    scale: EXPORT_SCALE
  }),
  description: 'S2_jambi_city_expanded_2024_20m',
  folder: EXPORT_FOLDER,
  scale: EXPORT_SCALE,
  region: jambiRegion,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

// Export region boundary for reference
Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(jambiRegion)]),
  description: 'jambi_city_expanded_boundary',
  folder: EXPORT_FOLDER,
  fileFormat: 'GeoJSON'
});

print('✓ Export tasks ready! Click "Run" in the Tasks tab.');

// ============================================================================
// COMPARISON STATISTICS
// ============================================================================

print('');
print('═══════════════════════════════════════════════');
print('COMPARISON STATISTICS');
print('═══════════════════════════════════════════════');
print('Legacy BBox area:', LEGACY_BBOX.area().divide(1e6), 'km²');
print('Expanded region area:', jambiRegion.area().divide(1e6), 'km²');
print('Expansion factor used:', EXPANSION_FACTOR);
print('Buffer distance:', bufferDistanceKm, 'km');
