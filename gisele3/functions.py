import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import *
from scipy.spatial import Delaunay
import rasterio
import networkx as nx
import gdal
import rasterio.mask
from osgeo import gdal
from sklearn.cluster import DBSCAN,OPTICS
def create_grid(crs,resolution,study_area):
    ''' This function creates a grid of points, returning a geopanda dataframe. The input is the following:
    crs -> The preffered crs of the dataframe (according to the case study), input should be an integer.
    resolution -> The preffered resolution of the grid of points, input should be an integer.
    study_area -> This is a shapely polygon, that has to be in the preffered crs.
    '''
    # crs and resolution should be a numbers, while the study area is a polygon
    df = pd.DataFrame(columns=['X', 'Y'])
    min_x=float(study_area.bounds[0])
    min_y=float(study_area.bounds[1])
    max_x=float(study_area.bounds[2])
    max_y = float(study_area.bounds[3])
    # create one-dimensional arrays for x and y
    lon = np.arange(min_x, max_x, resolution)
    lat = np.arange(min_y, max_y, resolution)
    lon, lat = np.meshgrid(lon, lat)
    print('Mesh created')
    df['X'] = lon.reshape((np.prod(lon.shape),))
    df['Y'] = lat.reshape((np.prod(lat.shape),))
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y),
                            crs=crs)
    print('Dataframe created')
    geo_df_clipped = gpd.clip(geo_df,study_area)
    #geo_df_clipped.to_file(r'Test\grid_of_points.shp')
    return geo_df_clipped

def delaunay_test(new_points,crs):
    tocki = new_points['geometry'].values
    number_points = new_points.shape[0]
    arr = np.zeros([number_points,2])
    counter=0
    for i in tocki:
        x = i.xy[0][0]
        y=i.xy[1][0]
        arr[counter,0] = x
        arr[counter,1] = y
        counter+=1
    tri = Delaunay(arr)
    triangle_sides = tri.simplices
    final_sides = []
    for i in triangle_sides:
        a=i[0]
        b=i[1]
        c=i[2]
        if a>b:
            final_sides.append((i[0],i[1]))
        else:
            final_sides.append((i[1], i[0]))
        if b>c:
            final_sides.append((i[1],i[2]))
        else:
            final_sides.append((i[2], i[1]))
        if a>c:
            final_sides.append((i[0],i[2]))
        else:
            final_sides.append((i[2], i[0]))
    final_sides2 = list(set(final_sides))
    new_lines=gpd.GeoDataFrame() # dataframe without the new possible connections
    print('Delaunay finished')
    new_points = new_points.reset_index()
    graph=nx.Graph()
    #feasible = [1 if (new_points.loc[new_points.index == i, 'geometry'].values[0]].distance(new_points.loc[new_points.index == j, 'geometry'].values[0]))>200 else 0 for i,j in final_sides2]
    count=0
    ID1 = []
    ID2= []
    LENGTH = []
    LINES = []
    for i, j in final_sides2:
        point1 = new_points.loc[new_points.index == i, 'geometry'].values[0]
        point2 = new_points.loc[new_points.index== j, 'geometry'].values[0]
        ID1.append(int(new_points.loc[new_points.index == i, 'ID'].values[0]))
        ID2.append(int(new_points.loc[new_points.index== j, 'ID'].values[0]))
        LENGTH.append(point1.distance(point2))
        LINES.append(LineString([point1, point2]))
        print('\r' + str(count) + '/' + str(len(final_sides)),
              sep=' ', end='', flush=True)
        count+=1
    [graph.add_edge(ID1[i],ID2[i],weight=LENGTH[i]) for i in range(len(ID1))]
    new_lines = gpd.GeoDataFrame({'ID1':ID1,'ID2':ID2,'length':LENGTH},geometry=LINES,crs=crs)

    return graph,new_lines

def process_population(population_location,study_area,crs):
    with rasterio.open(population_location,

                       mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area, crop=True)
        print(src.crs)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open('D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population\Population.tif', "w", **out_meta) as dest:
        dest.write(out_image)
    input_raster = gdal.Open('D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population\Population.tif')
    output_raster = 'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population\Population_crs.tif'
    warp = gdal.Warp(output_raster, input_raster, dstSRS='EPSG:'+str(crs))
    warp = None  # Closes the files

def run_DBSCAN(Population_gdf,min_eps,min_pop):
    loc = {'x': Population_gdf.geometry.x, 'y': Population_gdf.geometry.y}
    pop_points = pd.DataFrame(data=loc).values
    db = DBSCAN(eps=float(min_eps), min_samples=float(min_pop),
                metric='euclidean').fit(pop_points, sample_weight=Population_gdf['Population'])
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    Population_gdf['Clusters'] = labels
    percentage_clustered = Population_gdf.loc[Population_gdf['Clusters']!=-1,'Population'].sum()/Population_gdf['Population'].sum()*100
    return Population_gdf,n_clusters,percentage_clustered

def run_OPTICS(Population_gdf,min_pop):
    loc = {'x': Population_gdf.geometry.x, 'y': Population_gdf.geometry.y}
    pop_points = pd.DataFrame(data=loc).values
    db = OPTICS(min_samples=min_pop).fit(pop_points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    Population_gdf['Clusters'] = labels
    percentage_clustered = Population_gdf.loc[Population_gdf['Clusters']!=-1,'Population'].sum()/Population_gdf['Population'].sum()*100
    return Population_gdf,n_clusters,percentage_clustered

def rasterize_point_vector(point_vector,vector_resolution):
    #https://gis.stackexchange.com/questions/136059/convert-vector-point-shape-to-raster-tif-using-python-gdal-lib-in-qgis
    NoData_value = 0
    x_res = vector_resolution  # assuming these are the cell sizes
    y_res = vector_resolution  # change as appropriate
    pixel_size = 1

    # 2. Filenames for in- and output
    point_vector = r"D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia"
    output_raster = r"D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia_raster.tif"

    # 3. Open Shapefile
    source_ds = gpd.read_file(point_vector)
    x_min, x_max, y_min, y_max = GetExtent_point_layer(source_ds)

    # 4. Create Target - TIFF
    cols = int((x_max - x_min) / x_res)
    rows = int((y_max - y_min) / y_res)

    raster = gdal.GetDriverByName('GTiff').Create(_out, cols, rows, 1, gdal.GDT_Byte)
    _raster.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    _band = _raster.GetRasterBand(1)
    _band.SetNoDataValue(NoData_value)

    # 5. Rasterize why is the burn value 0... isn't that the same as the background?
    gdal.RasterizeLayer(_raster, [1], source_layer, burn_values=[0])

def GetExtent_point_layer(point_vector):

    x_coordinates = [row['geometry'].xy[0][0] for i, row in point_vector.iterrows()]
    y_coordinates = [row['geometry'].xy[1][0] for i, row in point_vector.iterrows()]
    max_x = max(x_coordinates)
    min_x = min(x_coordinates)
    max_y = max(y_coordinates)
    min_y = min(y_coordinates)

    print(max_x)
    print(min_x)
    print(max_y)
    print(min_y)
    return min_x, max_x, min_y, max_y
