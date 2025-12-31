// ============================================================================
// BOUNDARY VERIFICATION SCRIPT
// Run this FIRST before using the main analysis script!
// Purpose: Verify exact field values for administrative boundaries
// ============================================================================

// ============================================================================
// 1. CHECK GEOBOUNDARIES DATASET
// ============================================================================
print('========================================');
print('GEOBOUNDARIES v6.0 - Indonesia Check');
print('========================================');

var geoBoundaries = ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM1');

// Filter for Indonesia (ISO 3166-1 alpha-3 code)
var gbIndonesia = geoBoundaries.filter(ee.Filter.eq('shapeGroup', 'IDN'));

print('Total Indonesia provinces (geoBoundaries):', gbIndonesia.size());
print('All province names (shapeName field):');
print(gbIndonesia.aggregate_array('shapeName').sort());

// Check available fields
print('Available fields:', gbIndonesia.first().propertyNames());

// Try to find Jambi with different possible spellings
var possibleNames = ['Jambi', 'JAMBI', 'jambi', 'Provinsi Jambi'];
possibleNames.forEach(function(name) {
  var found = gbIndonesia.filter(ee.Filter.eq('shapeName', name));
  print('Searching "' + name + '":', found.size());
});

// ============================================================================
// 2. CHECK FAO GAUL DATASET
// ============================================================================
print('========================================');
print('FAO GAUL 2015 Level 1 - Indonesia Check');
print('========================================');

var faoGaul = ee.FeatureCollection('FAO/GAUL/2015/level1');

// Filter for Indonesia
var gaulIndonesia = faoGaul.filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'));

print('Total Indonesia provinces (FAO GAUL):', gaulIndonesia.size());
print('All province names (ADM1_NAME field):');
print(gaulIndonesia.aggregate_array('ADM1_NAME').sort());

// Check available fields
print('Available fields:', gaulIndonesia.first().propertyNames());

// Try to find Jambi
var gaulJambi = gaulIndonesia.filter(ee.Filter.eq('ADM1_NAME', 'Jambi'));
print('FAO GAUL "Jambi" found:', gaulJambi.size());

// ============================================================================
// 3. VISUALIZE BOTH DATASETS
// ============================================================================
print('========================================');
print('VISUALIZATION');
print('========================================');

// Center on Indonesia
Map.setCenter(117.0, -2.5, 5);

// Style for geoBoundaries
var gbStyle = {
  color: '0000FF',
  fillColor: '0000FF22',
  width: 2
};

// Style for FAO GAUL
var gaulStyle = {
  color: 'FF0000',
  fillColor: 'FF000022',
  width: 2
};

// Add layers
Map.addLayer(gbIndonesia.style(gbStyle), {}, 'geoBoundaries (Blue)');
Map.addLayer(gaulIndonesia.style(gaulStyle), {}, 'FAO GAUL (Red)');

// ============================================================================
// 4. COMPARE SPECIFIC PROVINCE (JAMBI)
// ============================================================================
print('========================================');
print('JAMBI PROVINCE COMPARISON');
print('========================================');

// Get Jambi from FAO GAUL (known to work)
var jambiGaul = gaulIndonesia.filter(ee.Filter.eq('ADM1_NAME', 'Jambi'));

if (jambiGaul.size().getInfo() > 0) {
  var jambiGeom = jambiGaul.geometry();
  print('Jambi (GAUL) area (km²):', jambiGeom.area().divide(1e6));

  // Zoom to Jambi
  Map.centerObject(jambiGeom, 8);

  // Highlight Jambi
  Map.addLayer(jambiGaul.style({color: '00FF00', fillColor: '00FF0044', width: 3}),
               {}, 'Jambi - FAO GAUL (Green)');
}

// ============================================================================
// 5. SUMMARY & RECOMMENDATIONS
// ============================================================================
print('========================================');
print('INSTRUCTIONS');
print('========================================');
print('1. Check the Console output above');
print('2. Note the exact spelling of province names');
print('3. Compare boundaries visually on the map');
print('4. Update PROVINCE_NAME in main script if needed');
print('');
print('Dataset Recommendations:');
print('- FAO GAUL: More reliable field names, well documented');
print('- geoBoundaries: Newer data (2023), open license (CC BY 4.0)');
