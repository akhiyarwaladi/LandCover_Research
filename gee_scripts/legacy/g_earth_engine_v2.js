// Mendefinisikan area Provinsi Jambi
var jambi = ee.Geometry.Rectangle([103.4486, -1.8337, 103.7566, -1.4089]);

// Fungsi untuk mask awan dan bayangan dengan metode yang diperbarui
function maskS2clouds(image) {
  var scl = image.select('SCL');
  
  // Identify clouds, shadows, and snow using SCL
  var cloudMask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9))  
    .and(scl.neq(1).and(scl.neq(2)));
    
  // Use cloud probability instead of QA60
  var cloudProb = image.select('MSK_CLDPRB');
  var cloudProbMask = cloudProb.lt(40);  // Mask pixels with >40% cloud probability
  
  // Add buffer around clouds
  var cloudBuffer = cloudMask.focal_min({
    radius: 50,
    kernelType: 'circle',
    units: 'meters'
  });
  
  // Combine masks and apply them
  var finalMask = cloudBuffer.and(cloudProbMask);
  
  return image.updateMask(finalMask)
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    .divide(10000);
}

// Mengambil data Sentinel-2 Surface Reflectance untuk 2024
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2024-05-01', '2024-09-30')
    .filterBounds(jambi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
    .map(maskS2clouds);

// Dynamic World dengan filtering yang sama
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate('2024-05-01', '2024-09-30')
    .filterBounds(jambi);

// Composite creation dengan quality metrics
var s2Composite = s2.median();
var dwComposite = dw.select('label').mode();

// Menambahkan probability bands dari Dynamic World
var dwProbability = dw.select([
  'water',
  'trees',
  'grass',
  'flooded_vegetation',
  'crops',
  'shrub_and_scrub',
  'built',
  'bare',
  'snow_and_ice'
]).mean();

// Post-processing untuk mengurangi noise
var kernel = ee.Kernel.circle({radius: 1});
var dwSmoothed = dwComposite.focal_mode({
  kernel: kernel,
  iterations: 2
});

// Calculate additional indices for validation
var indices = s2Composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    .addBands(s2Composite.normalizedDifference(['B8', 'B11']).rename('NDMI'))
    .addBands(s2Composite.normalizedDifference(['B3', 'B11']).rename('MNDWI'));

// Visualization parameters
var rgbVis = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.3,
  gamma: 1.2
};

var dwVis = {
  min: 0,
  max: 8,
  palette: [
    '#419BDF',    // water
    '#397D49',    // trees
    '#88B053',    // grass
    '#7A87C6',    // flooded vegetation
    '#E49635',    // crops
    '#DFC35A',    // shrub and scrub
    '#C4281B',    // built
    '#A59B8F',    // bare
    '#B39FE1'     // snow and ice
  ]
};

// Map layers
Map.centerObject(jambi, 11);
Map.addLayer(s2Composite, rgbVis, 'Sentinel-2 RGB');
Map.addLayer(dwSmoothed, dwVis, 'Dynamic World Classification (Smoothed)');
Map.addLayer(indices.select('NDVI'), 
  {min: -1, max: 1, palette: ['red', 'yellow', 'green']}, 
  'NDVI', false);

// Export Sentinel-2 (10m bands)
Export.image.toDrive({
  image: s2Composite.select(['B2', 'B3', 'B4', 'B8'])
    .reproject({crs: 'EPSG:4326', scale: 10}),
  description: 'sentinel2_jambi_2024_10m',
  folder: 'GEE_Exports',
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export Dynamic World classification (Byte type)
Export.image.toDrive({
  image: dwSmoothed.reproject({crs: 'EPSG:4326', scale: 10}),
  description: 'dynamicworld_jambi_2024_classification',
  folder: 'GEE_Exports',
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export Dynamic World probabilities (Float type)
Export.image.toDrive({
  image: dwProbability.reproject({crs: 'EPSG:4326', scale: 10}),
  description: 'dynamicworld_jambi_2024_probability',
  folder: 'GEE_Exports',
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});

// Export indices
Export.image.toDrive({
  image: indices.reproject({crs: 'EPSG:4326', scale: 10}),
  description: 'indices_jambi_2024',
  folder: 'GEE_Exports',
  scale: 10,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {cloudOptimized: true}
});