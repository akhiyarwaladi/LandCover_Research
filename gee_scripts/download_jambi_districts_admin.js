// ============================================================================
// Download Sentinel-2 for Jambi City Administrative Districts
// ============================================================================
//
// Uses NATURAL administrative boundaries (kabupaten/kota level) instead of
// circular or rectangular crops. This gives beautiful curved boundaries!
//
// Author: Claude Sonnet 4.5
// Date: 2026-01-03
// ============================================================================

// ============================================================================
// STEP 1: Load Administrative Boundaries (Level 2 - Districts)
// ============================================================================

// Load all level 2 admin boundaries in Jambi Province
var jambiDistricts = ee.FeatureCollection("FAO/GAUL/2015/level2")
    .filter(ee.Filter.eq('ADM1_NAME', 'Jambi'));

// Print available districts
print('Jambi Districts (Kabupaten/Kota):', jambiDistricts.aggregate_array('ADM2_NAME').distinct());
print('Total districts:', jambiDistricts.size());

// Visualize all districts
Map.centerObject(jambiDistricts, 9);
Map.addLayer(jambiDistricts, {color: 'lightgray'}, 'All Jambi Districts', false);

// ============================================================================
// STEP 2: Select Districts Around Jambi City
// ============================================================================

// Option 1: Just Jambi City (smallest - likely too small)
var jambiCityOnly = jambiDistricts
    .filter(ee.Filter.or(
        ee.Filter.eq('ADM2_NAME', 'Jambi'),
        ee.Filter.eq('ADM2_NAME', 'Kota Jambi')
    ));

// Option 2: Jambi City + Muaro Jambi (moderate size - RECOMMENDED)
var jambiCityPlus1 = jambiDistricts
    .filter(ee.Filter.or(
        ee.Filter.eq('ADM2_NAME', 'Jambi'),
        ee.Filter.eq('ADM2_NAME', 'Kota Jambi'),
        ee.Filter.eq('ADM2_NAME', 'Muaro Jambi')
    ));

// Option 3: Jambi City + 2 adjacent districts (larger)
var jambiCityPlus2 = jambiDistricts
    .filter(ee.Filter.or(
        ee.Filter.eq('ADM2_NAME', 'Jambi'),
        ee.Filter.eq('ADM2_NAME', 'Kota Jambi'),
        ee.Filter.eq('ADM2_NAME', 'Muaro Jambi'),
        ee.Filter.eq('ADM2_NAME', 'Batanghari'),
        ee.Filter.eq('ADM2_NAME', 'Batang Hari')
    ));

// ============================================================================
// SELECT YOUR PREFERRED OPTION HERE:
// ============================================================================

var selectedRegion = jambiCityPlus1;  // 👈 CHANGE THIS to try different options
var regionName = 'Jambi_City_Plus_1';  // 👈 CHANGE THIS to match

print('Selected region:', selectedRegion.aggregate_array('ADM2_NAME'));
print('Number of districts:', selectedRegion.size());

// Visualize selected districts
Map.addLayer(jambiCityOnly, {color: 'yellow'}, 'Option 1: City Only', false);
Map.addLayer(jambiCityPlus1, {color: 'orange'}, 'Option 2: City + 1 District (RECOMMENDED)', true);
Map.addLayer(jambiCityPlus2, {color: 'red'}, 'Option 3: City + 2 Districts', false);

// Get unified geometry with natural boundaries
var jambiRegion = selectedRegion.geometry();

// Add legacy bbox for comparison
var LEGACY_BBOX = ee.Geometry.Rectangle([103.4486, -1.8337, 103.7566, -1.4089]);
Map.addLayer(LEGACY_BBOX, {color: 'blue'}, 'Legacy BBox (Reference)', false);

// Calculate area
var areaKm2 = jambiRegion.area().divide(1e6);
print('Selected region area (km²):', areaKm2);
print('Region bounds:', jambiRegion.bounds());

// ============================================================================
// STEP 3: Cloud Masking Function
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
// STEP 4: Load and Process Sentinel-2 Data
// ============================================================================

var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2024-01-01', '2024-12-31')
    .filterBounds(jambiRegion)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(maskS2clouds);

var s2Composite = s2.median();

// ============================================================================
// STEP 5: Visualization
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
// STEP 6: Export
// ============================================================================

var EXPORT_SCALE = 20;  // 20m resolution
var EXPORT_FOLDER = 'GEE_Exports';
var EXPORT_CRS = 'EPSG:4326';

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
  description: 'S2_' + regionName + '_2024_20m',
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
  collection: selectedRegion,
  description: regionName + '_boundary',
  folder: EXPORT_FOLDER,
  fileFormat: 'GeoJSON'
});

print('✓ Export tasks ready! Click "Run" in the Tasks tab.');

// ============================================================================
// COMPARISON STATISTICS
// ============================================================================

print('');
print('═══════════════════════════════════════════════');
print('AREA COMPARISON');
print('═══════════════════════════════════════════════');
print('Legacy BBox area:', LEGACY_BBOX.area().divide(1e6), 'km²');
print('Selected region area:', areaKm2, 'km²');
print('Districts included:', selectedRegion.aggregate_array('ADM2_NAME'));
print('');
print('💡 TIP: Change "selectedRegion" and "regionName" at the top');
print('   to try different district combinations!');
