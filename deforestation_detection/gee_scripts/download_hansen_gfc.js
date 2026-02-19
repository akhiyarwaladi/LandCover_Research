/**
 * Download Hansen Global Forest Change (GFC) Data for Jambi Province
 *
 * Downloads 3 layers from Hansen et al. (2013) GFC v1.12:
 *   1. treecover2000 - Tree canopy cover in year 2000 (%)
 *   2. lossyear - Year of forest loss (1-24 = 2001-2024)
 *   3. gain - Forest gain 2000-2012 (binary)
 *
 * Native resolution: 30m (will be resampled to 20m in preprocessing)
 *
 * Usage: Paste into Google Earth Engine Code Editor and run.
 *
 * Reference:
 *   Hansen, M.C., et al. (2013). High-Resolution Global Maps of
 *   21st-Century Forest Cover Change. Science, 342(6160), 850-853.
 */

// ============================================================
// 1. Define Study Area
// ============================================================

var jambi = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM1_NAME', 'Jambi'));

var jambiBounds = jambi.geometry();

Map.centerObject(jambiBounds, 8);
Map.addLayer(jambiBounds, {color: 'red'}, 'Jambi Province');

// ============================================================
// 2. Load Hansen GFC Dataset
// ============================================================

var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');

// Extract layers
var treecover2000 = gfc.select('treecover2000').clip(jambiBounds);
var lossyear = gfc.select('lossyear').clip(jambiBounds);
var gain = gfc.select('gain').clip(jambiBounds);

// ============================================================
// 3. Visualize
// ============================================================

// Tree cover 2000
Map.addLayer(treecover2000, {min: 0, max: 100, palette: ['white', 'green']},
             'Tree Cover 2000', false);

// Loss year (red = recent, blue = old)
var lossVis = {min: 1, max: 24, palette: [
  '#0000FF', '#0033FF', '#0066FF', '#0099FF', '#00CCFF',
  '#00FFCC', '#00FF99', '#00FF66', '#00FF33', '#00FF00',
  '#33FF00', '#66FF00', '#99FF00', '#CCFF00', '#FFFF00',
  '#FFCC00', '#FF9900', '#FF6600', '#FF3300', '#FF0000',
  '#CC0000', '#990000', '#660000', '#330000'
]};
Map.addLayer(lossyear.updateMask(lossyear.gt(0)), lossVis,
             'Loss Year (2001-2024)', true);

// Forest gain
Map.addLayer(gain.updateMask(gain.gt(0)), {palette: ['blue']},
             'Forest Gain 2000-2012', false);

// ============================================================
// 4. Compute Statistics
// ============================================================

// Total forest area in 2000 (>30% tree cover)
var forest2000 = treecover2000.gte(30);
var forestArea = forest2000.multiply(ee.Image.pixelArea()).reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: jambiBounds,
  scale: 30,
  maxPixels: 1e13
});
print('Forest area in 2000 (m2):', forestArea);

// Loss per year (2018-2024 = values 18-24)
for (var yr = 18; yr <= 24; yr++) {
  var yearLoss = lossyear.eq(yr);
  var lossArea = yearLoss.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: jambiBounds,
    scale: 30,
    maxPixels: 1e13
  });
  print('Loss area ' + (2000 + yr) + ' (m2):', lossArea);
}

// ============================================================
// 5. Create Derived Layers
// ============================================================

// Forest mask at 30% threshold
var forestMask = treecover2000.gte(30).rename('forest_mask_2000');

// Loss in our study period (2018-2024)
var lossStudyPeriod = lossyear.gte(18).and(lossyear.lte(24))
    .rename('loss_2018_2024');

// Remaining forest (forest2000 minus all losses through 2024)
var allLoss = lossyear.gt(0);
var remainingForest = forest2000.and(allLoss.not()).rename('remaining_forest');

// Combine all layers
var hansenStack = treecover2000
    .addBands(lossyear)
    .addBands(gain)
    .addBands(forestMask)
    .addBands(lossStudyPeriod)
    .addBands(remainingForest)
    .toFloat();

// Visualize study period loss
Map.addLayer(lossStudyPeriod.updateMask(lossStudyPeriod),
             {palette: ['red']}, 'Loss 2018-2024', false);
Map.addLayer(remainingForest.updateMask(remainingForest),
             {palette: ['darkgreen']}, 'Remaining Forest', false);

// ============================================================
// 6. Export to Google Drive
// ============================================================

// Export treecover2000
Export.image.toDrive({
  image: treecover2000.toFloat(),
  description: 'Hansen_treecover2000_Jambi',
  folder: 'GEE_Deforestation_Jambi',
  fileNamePrefix: 'Hansen_treecover2000_Jambi',
  region: jambiBounds,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export lossyear
Export.image.toDrive({
  image: lossyear.toFloat(),
  description: 'Hansen_lossyear_Jambi',
  folder: 'GEE_Deforestation_Jambi',
  fileNamePrefix: 'Hansen_lossyear_Jambi',
  region: jambiBounds,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export gain
Export.image.toDrive({
  image: gain.toFloat(),
  description: 'Hansen_gain_Jambi',
  folder: 'GEE_Deforestation_Jambi',
  fileNamePrefix: 'Hansen_gain_Jambi',
  region: jambiBounds,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// Export combined stack
Export.image.toDrive({
  image: hansenStack,
  description: 'Hansen_GFC_stack_Jambi',
  folder: 'GEE_Deforestation_Jambi',
  fileNamePrefix: 'Hansen_GFC_stack_Jambi',
  region: jambiBounds,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

// ============================================================
// 7. Summary
// ============================================================

print('=== Hansen GFC Download for Jambi ===');
print('Dataset: UMD/hansen/global_forest_change_2024_v1_12');
print('Layers: treecover2000, lossyear, gain + derived');
print('Resolution: 30m (native)');
print('Tree cover threshold: 30%');
print('Study period: 2018-2024 (lossyear values 18-24)');
print('');
print('Click RUN on each export task in the Tasks tab.');
print('Files will be saved to Google Drive: GEE_Deforestation_Jambi/');
