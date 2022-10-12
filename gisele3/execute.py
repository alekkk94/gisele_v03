import geopandas as gpd
import rasterio
from gisele3.functions import *
import os
import time
def process_population():
    crs = 32736
    resolution=100
    admin_folder = r'Admin'
    regional_division_file = r'gadm36_MOZ_1.shp'
    mozambique_folder = r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\_vania_2.0\Data\Case_Study\Country Level\Mozambique'
    regions = gpd.read_file(os.path.join(mozambique_folder,admin_folder,regional_division_file))
    regions_crs=regions.to_crs(crs)
    zambezia_polygon = regions.loc[regions['NAME_1']=='Zambezia','geometry'].values[0][45] #the mainland
    zambezia_polygon_crs = regions_crs.loc[regions_crs['NAME_1']=='Zambezia','geometry'].values[0][45] #the mainland
    #new_lines=gpd.read_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\delaunay')
    #new_lines.set_crs(32763)
    #new_lines[new_lines['length']<1].to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\delaunay2')
    population_Raster =rasterio.open(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population\Population_crs.tif')
    pointData = create_grid(crs,resolution,zambezia_polygon_crs)
    print('Grid of points created.')
    coords = [(x, y) for x, y in zip(pointData.X, pointData.Y)]
    pointData = pointData.reset_index(drop=True)
    #pointData['ID'] = pointData.index
    pointData['Population'] = [x[0]for x in population_Raster.sample(coords)]
    print('Population values assigned')
    pointData.drop(['X','Y'],axis=1,inplace=True)
    Population_gdf = pointData[pointData['Population']>0]

    Population_gdf.to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia')
start=time.time()
process_population()
end=time.time()
print('Time required is '+str(end-start) +' seconds.')