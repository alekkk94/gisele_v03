import geopandas as gpd
import pandas as pd
import os
import time
from osgeo import gdal, ogr, osr
#https://www.youtube.com/watch?v=lQW9zK79M80
start=time.time()
point_vector = r"D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia"
output_raster = r"D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia_raster.tif"

source_ds = gpd.read_file(point_vector)

x_res=100
y_res=100
NoData_value=0
max_x=1153592.729
min_x=729392.729
max_y = 8252422.529
min_y = 7914512.529
cols = int((max_x - min_x) / x_res)
rows = int((max_y - min_y) / y_res)

input_shp = ogr.Open(point_vector)
source_layer = input_shp.GetLayer()

raster = gdal.GetDriverByName('GTiff').Create(output_raster, cols, rows, 1, gdal.GDT_Byte)
raster.SetGeoTransform((min_x, x_res, 0, max_y, 0, -y_res))
band = raster.GetRasterBand(1)
band.SetNoDataValue(NoData_value)

gdal.RasterizeLayer(raster, [1], source_layer, burn_values=[255],options = ['ALL_TOUCHED=FALSE'])
new_rasterSRS = osr.SpatialReference()
new_rasterSRS.ImportFromEPSG(32736)
raster.SetProjection(new_rasterSRS.ExportToWkt())


#NEW
point_vector = r"D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia"
pixel_size=100
NoData_value=0

output_raster = r"D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia_raster.tif"
input_shp = ogr.Open(point_vector)
source_layer = input_shp.GetLayer()
#min_x,max_x,min_y,max_y = source_layer.GetExtent()
max_x=1153592.729
min_x=729392.729
max_y = 8252422.529
min_y = 7914512.529
x_res = int((max_x - min_x) / pixel_size)
y_res = int((max_y - min_y) / pixel_size)
defn=source_layer.GetLayerDefn()
column_names=[]
for n in range(defn.GetFieldCount()):
    fdefn=defn.GetFieldDefn(n)
    column_names.append(fdefn.name)
print(column_names)


target_ds = gdal.GetDriverByName('GTiff').Create(output_raster,x_res,y_res,1,gdal.GDT_Float32,['COMPRESS=LZW'])
target_ds.SetGeoTransform((min_x, pixel_size, 0.0, max_y, 0.0, -pixel_size))
sr = osr.SpatialReference()
sr.SetProjection ('EPSG:32736')
sr_wkt = sr.ExportToWkt()
target_ds.SetProjection(sr_wkt)
band = target_ds.GetRasterBand(1)
target_ds.GetRasterBand(1).SetNoDataValue(NoData_value)
#band.fill(NoData_value)
gdal.RasterizeLayer(target_ds, [1], source_layer, None,None, [1],options = ['ALL_TOUCHED=TRUE' , 'ATTRIBUTE=Population'])
target_ds.GetRasterBand(1).SetNoDataValue(NoData_value)
target_ds = None



