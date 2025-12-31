// Mendefinisikan area Provinsi Jambi
var jambi = ee.Geometry.Rectangle([103.4486, -1.8337, 103.7566, -1.4089]);

// Fungsi untuk mask awan dan bayangan dengan menggunakan MSK bands yang baru
function maskS2clouds(image) {
  var scl = image.select('SCL');
  var cloudMask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9))  // Remove cloud, cloud shadow, and snow
    .and(scl.neq(1).and(scl.neq(2)));  // Remove saturated and defective pixels
    
  // Menggunakan MSK bands sebagai pengganti QA60
  var cloudProb = image.select('MSK_CLDPRB');
  var opaque = image.select('MSK_CLASSI_OPAQUE');
  var cirrus = image.select('MSK_CLASSI_CIRRUS');
  
  // Membuat mask berdasarkan probabilitas awan dan klasifikasi
  var cldProbMask = cloudProb.lt(40);  // Mask piksel dengan probabilitas awan > 40%
  var opaqueMask = opaque.neq(1);      // Mask piksel yang terklasifikasi sebagai awan tebal
  var cirrusMask = cirrus.neq(1);      // Mask piksel yang terklasifikasi sebagai awan tipis
      
  // Combine masks and apply them
  var finalMask = cloudMask.and(cldProbMask).and(opaqueMask).and(cirrusMask);
  return image.updateMask(finalMask)
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    .divide(10000);  // Scale pixel values to 0-1
}

// Mengambil data Sentinel-2 Surface Reflectance
var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
    .filterDate('2024-01-01', '2024-12-31')
    .filterBounds(jambi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  // Lebih ketat untuk cloud filtering
    .map(maskS2clouds);

// Mengambil Dynamic World dengan filtering temporal yang sama
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    .filterDate('2024-01-01', '2024-12-31')
    .filterBounds(jambi);

// Membuat composite dengan median untuk mengurangi noise
var s2Composite = s2.median();
var dwComposite = dw.select('label').mode();

// Menambahkan visualisasi untuk preview
var rgbVis = {
  bands: ['B4', 'B3', 'B2'],
  min: 0,
  max: 0.3,
  gamma: 1.2
};

// Membuat layer untuk preview
Map.centerObject(jambi, 8);
Map.addLayer(s2Composite, rgbVis, 'Sentinel-2 RGB');
Map.addLayer(dwComposite, {}, 'Dynamic World Classification');

// NDVI calculation untuk quality check
var ndvi = s2Composite.normalizedDifference(['B8', 'B4']);
Map.addLayer(ndvi, {min: -1, max: 1, palette: ['red', 'yellow', 'green']}, 'NDVI');

// Export dengan resolusi 20m
var scale = 20;

// Export Sentinel-2
Export.image.toDrive({
  image: s2Composite.select([
    'B2', 'B3', 'B4',  // RGB
    'B5', 'B6', 'B7',  // Red Edge
    'B8', 'B8A',       // NIR
    'B11', 'B12'       // SWIR
  ]).reproject({
    crs: 'EPSG:4326',
    scale: scale
  }),
  description: 'sentinel2_jambi_2024_Q4_allbands',
  folder: 'GEE_Exports',
  scale: scale,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

// Export Dynamic World
Export.image.toDrive({
  image: dwComposite.reproject({
    crs: 'EPSG:4326',
    scale: scale
  }),
  description: 'dynamicworld_jambi_2024_Q4',
  folder: 'GEE_Exports',
  scale: scale,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});

// Export NDVI untuk QC
Export.image.toDrive({
  image: ndvi.reproject({
    crs: 'EPSG:4326',
    scale: scale
  }),
  description: 'ndvi_jambi_2024_Q4',
  folder: 'GEE_Exports',
  scale: scale,
  region: jambi,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true
  }
});