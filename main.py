from gisele3.functions import *
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import *
import rasterio
import pandas as pd

crs = 32736
mozambique_folder = r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\_vania_2.0\Data\Case_Study\Country Level\Mozambique'
#processing the population
buildings_folder = 'Buildings'
admin_folder = r'Admin'
pop_folder= 'Population'

buildings_file_name = 'buildings.tif'
pop_file_name = 'population_worldpop.tif'
regional_division_file = r'gadm36_MOZ_1.shp'

#population_Raster=rasterio.open(os.path.join(mozambique_folder,pop_folder,pop_file_name))
#crs = population_Raster.crs

regions = gpd.read_file(os.path.join(mozambique_folder,admin_folder,regional_division_file))
regions_crs=regions.to_crs(crs)
zambezia_polygon = regions.loc[regions['NAME_1']=='Zambezia','geometry'].values[0][45] #the mainland
zambezia_polygon_crs = regions_crs.loc[regions_crs['NAME_1']=='Zambezia','geometry'].values[0][45] #the mainland

resolution=100
#study_area = MultiPolygon([poly for poly in area.geometry])
#study_area = area.loc[area['NAME_3']=='Mulevala (Namigonha)','geometry'].values[0]
#affine = population_Raster.transform
#resolution = affine[0]
pointData = create_grid(crs,resolution,zambezia_polygon_crs)
process_population(os.path.join(mozambique_folder,pop_folder,pop_file_name),[zambezia_polygon],crs)
population_Raster =rasterio.open(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population\Population_crs.tif')
coords = [(x, y) for x, y in zip(pointData.X, pointData.Y)]
pointData = pointData.reset_index(drop=True)
pointData['ID'] = pointData.index
pointData['Population'] = [x[0]for x in population_Raster.sample(coords)]
pointData.to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population\population_vector')
#pointData.loc[pointData['Population']==65535,'Population']=0
#Buildings = pointData[pointData['Population']!=65535]
#Buildings.to_file(os.path.join('D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia','Population'))

#terrain = rasterio.open(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\_vania_2.0\Data\Case_Study\Country Level\Mozambique\LandCover\landcover.tif')
##terrain.transform[0]
#pointData['LandCover'] = [x[0]for x in terrain.sample(coords)]
#pointData.to_file(os.path.join('D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia','grid_points'))

process_population(os.path.join(mozambique_folder,pop_folder,pop_file_name),[study_area],crs)

#Buildings_crs = Buildings.to_crs(32737)
#graph,lines = delaunay_test(Buildings_crs)
#len(lines)
#len(lines[lines['length']<0.2])
#lines_200 = lines[lines['length']<0.2]
#lines_200
#lines_200.to_file(os.path.join('D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia','lines_200m'))