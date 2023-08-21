import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import *
import shapely
from scipy.spatial import Delaunay
import rasterio
import networkx as nx
#import gdal
import rasterio.mask
from osgeo import gdal,osr,ogr
from sklearn.cluster import DBSCAN,OPTICS,KMeans
import os
from geovoronoi import voronoi_regions_from_coords, points_to_coords
from scipy.spatial import cKDTree as KDTree
from scipy import sparse
import sys
import time
import math
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import requests
#from gisele3.michele.michele import start
import json
from collections import Counter
def create_grid(crs,resolution,study_area,return_lines=True):
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

def delaunay_test(new_points,crs,return_lines=True):
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
    
    print('Delaunay finished')
   
    graph=nx.Graph()
    #feasible = [1 if (new_points.loc[new_points.index == i, 'geometry'].values[0]].distance(new_points.loc[new_points.index == j, 'geometry'].values[0]))>200 else 0 for i,j in final_sides2]
    if return_lines:
        new_points = new_points.reset_index()
        new_lines=gpd.GeoDataFrame() # dataframe without the new possible connections
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
        new_lines = gpd.GeoDataFrame({'ID1':ID1,'ID2':ID2,'length':LENGTH},geometry=LINES,crs=crs)
        [graph.add_edge(ID1[i],ID2[i],weight=LENGTH[i]) for i in range(len(ID1))]
        
    
        return graph,new_lines
    else:
        id1 = [i for i,j in final_sides2]
        id2 = [j for i,j in final_sides2]
        df1 = new_points.loc[id1]
        df2 = new_points.loc[id2]
        loc1 = [[pt.xy[0][0],pt.xy[1][0]] for pt in df1.geometry]
        loc2 = [[pt.xy[0][0],pt.xy[1][0]] for pt in df2.geometry]
        start=time.time()
        length = [math.dist(loc1[i],loc2[i]) for i in range(len(loc1))]
        end=time.time()
        new_lines = pd.DataFrame({'ID1':id1,'ID2':id2,'length':length})
        [graph.add_edge(id1[i],id2[i],weight=length[i]) for i in range(len(id1))]
        
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

def run_k_means(Population_gdf,n_clusters,location_folder,crs,load_capita):
    # Starting k-means clustering

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0, max_iter=10000)

    # Running k-means clustering and enter the ‘X’ array as the input coordinates and ‘Y’ array as sample
    #     weights
    loc = {'x': Population_gdf.geometry.x, 'y': Population_gdf.geometry.y}
    pop_points = pd.DataFrame(data=loc).values
    population = Population_gdf.Population.values.astype(int)
    wt_kmeansclus = kmeans.fit(pop_points, sample_weight=population)
    predicted_kmeans = kmeans.predict(pop_points, sample_weight=population)

    # Storing results obtained together with respective city-state labels
    Population_gdf['Cl_Kmeans'] = predicted_kmeans
    centers = kmeans.cluster_centers_
    list_centroids = [Point(point[0],point[1]) for point in centers]
    centroids = gpd.GeoDataFrame({'cluster':[*range(0,len(centers))]},geometry = list_centroids,crs=crs)
    centroids['Population'] = [int(Population_gdf.loc[Population_gdf['Cl_Kmeans']==row['cluster'],'Population'].sum()) for i,row in centroids.iterrows()]
    centroids['power'] = [people * load_capita/1000 for people in centroids.Population]
    centroids.to_file(os.path.join(location_folder,'Centroids_kmeans'))

    Population_gdf.to_file(os.path.join(location_folder, 'Population_kmeans'))

    # Printing count of points alloted to each cluster and then the cluster centers
    #kmeans_results =pd.DataFrame({"label": data_label, "kmeans_cluster": predicted_kmeans + 1})
    #print(predicted_kmeans.kmeans_cluster.value_counts())
    return Population_gdf,centroids
def rasterize_point_vector(point_vector,vector_resolution):
    #https://gis.stackexchange.com/questions/136059/convert-vector-point-shape-to-raster-tif-using-python-gdal-lib-in-qgis
    #https://www.youtube.com/watch?v=lQW9zK79M80
    point_vector = r"C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia\Population_Zambezia"
    output_raster = r"C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia\Population_Zambezia_raster3.tif"

    # NEW

    pixel_size = 100
    NoData_value = 0

    input_shp = ogr.Open(point_vector)
    source_layer = input_shp.GetLayer()
    # min_x,max_x,min_y,max_y = source_layer.GetExtent()
    max_x = 1153592.729
    min_x = 729392.729
    max_y = 8342422.529
    min_y = 7914512.529
    x_res = int((max_x - min_x) / pixel_size)
    y_res = int((max_y - min_y) / pixel_size)
    defn = source_layer.GetLayerDefn()
    column_names = []
    for n in range(defn.GetFieldCount()):
        fdefn = defn.GetFieldDefn(n)
        column_names.append(fdefn.name)
    print(column_names)

    target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, x_res, y_res, 1, gdal.GDT_Float32, ['COMPRESS=LZW'])
    target_ds.SetGeoTransform((min_x, pixel_size, 0.0, max_y, 0.0, -pixel_size))
    sr = osr.SpatialReference()
    sr.SetWellKnownGeogCS('EPSG:32736')
    sr_wkt = sr.ExportToWkt()
    target_ds.SetProjection(sr_wkt)
    band = target_ds.GetRasterBand(1)
    target_ds.GetRasterBand(1).SetNoDataValue(NoData_value)
    target_ds.GetRasterBand(1).Fill(NoData_value)
    gdal.RasterizeLayer(target_ds, [1], source_layer, None, None, [1],
                        options=['ALL_TOUCHED=TRUE', 'ATTRIBUTE=Population'])
    target_ds.GetRasterBand(1).SetNoDataValue(NoData_value)
    target_ds = None

def polygon_around_points(points):
    'The function takes a geodataframe of points and returns a simple polygon'
    return points.unary_union.convex_hull


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

def process_schools_and_hospitals(population_points,hospitals,schools,area):
    database = r'C:\Users\alekd\Politecnico di Milano\Documentale DENG - Mozambique'
    output_folder = r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia'
    #Step 1 - transfer to the correct crs and clip
    #Step 2 - if there are schools that are too close to eachother -> merge
    #Step 3 - do the voronoi and count the population inside each polygon

    #In the final procedure, perhaps we create an energy map that considers the surounding as well (cleaned/filtered/averaged, whatever the name, moving average)

    schools = gpd.read_file(os.path.join(database,'Schools'))
    hospitals = gpd.read_file(os.path.join(database, 'Hospitals'))
    admin = gpd.read_file(os.path.join(database,'Admin','gadm36_MOZ_1.shp'))
    schools = schools.to_crs('EPSG:32736')
    hospitals = hospitals.to_crs('EPSG:32736')
    admin = admin.to_crs('EPSG:32736')
    area = admin.loc[admin['NAME_1']=='Zambezia','geometry'].values[0]
    if not type(area) == shapely.geometry.polygon.Polygon:
        areas = [polygon.area for polygon in area]
        index = areas.index(max(areas))
        area=area[index]

    schools_clipped=gpd.clip(schools,area)
    hospitals_clipped = gpd.clip(hospitals,area)
    hospitals_polygons = voronoi(hospitals_clipped,area,crs=32736)
    hospitals_polygons.to_file(os.path.join(output_folder,'hospitals_polygons'))
    schools_polygons = voronoi(hospitals_clipped, area, crs=32736)
    schools_polygons.to_file(os.path.join(output_folder, 'schools_polygons'))

    pass

def voronoi(points,area,crs):
    coords_PS = points_to_coords(points.geometry)
    a, b = voronoi_regions_from_coords(coords_PS, area)
    polygons = []
    primary_substations = []
    for key in list(a.keys()):
        polygon = a.get(key)
        if not type(polygon) == shapely.geometry.polygon.Polygon:
            print('Not a polygon')
        try:
            substation_inside_polygon = [polygon.contains(point['geometry']) for ii, point in
                                         points.iterrows()]
            ID_substation = points.loc[substation_inside_polygon.index(True), 'OBJECTID']
        except:
            ID_substation=99999
        primary_substations.append(ID_substation)
        polygons.append(polygon)

    voronoi_polygons = gpd.GeoDataFrame({'PS': primary_substations}, geometry=polygons, crs=crs)

    return voronoi_polygons

def point_density(point_gdf, calculate_percentiles=True, radius=284):
    '''Much more efficient method with sparse matrix, logarithmic rise'''
    n = len(point_gdf)
    loc = [[point_gdf.loc[i, 'geometry'].xy[0][0], point_gdf.loc[i, 'geometry'].xy[1][0]] for i in range(n)]
    # dist_matrix2 = scipy.spatial.distance_matrix(loc,loc)
    A = KDTree(loc)
    #end = time.time()
    # list all pairs within 0.05 of each other in 2-norm
    # format: (i, j, v) - i, j are indices, v is distance
    D = A.sparse_distance_matrix(A, radius, p=2.0, output_type='ndarray')

    # only keep upper triangle
    DU = D[D['i'] < D['j']]

    # make sparse matrix
    result = sparse.coo_matrix((DU['v'], (DU['i'], DU['j'])), (n, n))
    result_csr = result.tocsr()
    sparse_dist_matrix = [((i, j), result_csr[i, j]) for i, j in zip(*result_csr.nonzero())]
    a = [[] for x in range(n)]
    for i in range(len(sparse_dist_matrix)):
        point1 = sparse_dist_matrix[i][0][0]
        point2 = sparse_dist_matrix[i][0][1]
        a[point1].append(point2)
        a[point2].append(point1)

    population = point_gdf['Population'].tolist()
    new_pop_density_column = []
    
    new_pop_density_column = [sum(population[i] for i in a[counter])+population[counter] for counter in range(n)]
    

    point_gdf['Pop_'+str(radius)] = new_pop_density_column
    if calculate_percentiles:
        sorted_density = np.argsort(new_pop_density_column)
        sorted_population = np.sort(population)
        new_pop_density_sorted_percentile = np.zeros(n)
        percentile_population = np.zeros(n)
        new_pop_density_sorted = np.zeros(n)
        total_population = sum(population)
        sorted_population_cumsum = np.cumsum(sorted_population)
        for i in range(n):
            point = sorted_density[i]
            new_pop_density_sorted[point] = i
            new_pop_density_sorted_percentile[point] = i / n * 100
            percentile_population[point] = sorted_population_cumsum[
                                               i] / total_population * 100  # this is considers the percentile as well. In the end, perhaps the 10%
            # most densely populated points will have 30% of the population. This is quite important for the clustering and sensitivity analysis.
            #print('\r' + str(i) + '/' + str(n),
            #      sep=' ', end='', flush=True)
    
        point_gdf['perce_pts'] = new_pop_density_sorted_percentile  # this is percentiles but just in terms of points, it doesn't consider the population
        point_gdf['perce_ppl'] = percentile_population

    return point_gdf

def novel_clustering(points_gdf):
    pass

def process_google_buildings(buildings,area,crs):
    import shapely.wkt
    buildings['geometry'] = gpd.GeoSeries.from_wkt(buildings['geometry'])
    buildings_gdf = gpd.GeoDataFrame(buildings, geometry='geometry',crs=4326)
    buildings_gdf.to_crs('EPSG:'+str(crs),inplace=True)
    buildings_gdf.drop(columns=['latitude','longitude','full_plus_code'],inplace=True)
    buildings_gdf_centroids = buildings_gdf.copy()
    buildings_gdf_centroids.geometry = buildings_gdf_centroids.geometry.centroid
    final_buildings = gpd.sjoin(buildings_gdf_centroids,area,how='inner',op='within')
    final_buildings.drop(columns=['index_right'],inplace=True)
    final_buildings.to_file(r"C:\Users\alekd\OneDrive - Politecnico di Milano\PhD studies\Papers\gisele-3 paper\Data\Zambezia\Buildings_google.gpkg",driver="GPKG")
    

def connection(all_points_gdf,point1,point2,direct_connection=False):
    '''This script creates a conneciton between point1 and point2, following the grid of points all_points_gdf'''
    
    
    dist = point1.geometry.values[0].distance(point2.geometry.values[0])
    if dist < 1000:
        extension = dist
    elif dist < 5000:
        extension = dist * 0.6
    else:
        extension = dist / 5
    
    x_min = min(point1.geometry.values[0].coords[0][0],point2.geometry.values[0].coords[0][0])
    x_max = max(point1.geometry.values[0].coords[0][0],point2.geometry.values[0].coords[0][0])
    y_min = min(point1.geometry.values[0].coords[0][1],point2.geometry.values[0].coords[0][1])
    y_max = max(point1.geometry.values[0].coords[0][1],point2.geometry.values[0].coords[0][1])
    bubble = box(minx=x_min - extension, maxx=x_max + extension,
                 miny=y_min - extension, maxy=y_max + extension)
    area=gpd.GeoDataFrame(geometry=[bubble],crs=crs)
    points = gpd.sjoin(all_points_gdf,area,how="inner",predicate="within")
    points = points[list(all_points_gdf.columns)]
    c_grid_points=[]
    line_bc=10000
    resolution=200
    Rivers_option=False
    #make sure that point1 and point2 have weight,X and Y
    
    connection_gdf = dijkstra_connection(points, point1, point2,
                            c_grid_points, line_bc,resolution)

    
    
   # Lines_connections = gpd.GeoDataFrame(Lines_connections, geometry=Lines_connections['geometry'], crs=crs)
    #Lines_connections.to_file(data_folder + '/Lines_connections')
    #Lines_connections.to_csv(data_folder + '/Lines_connections.csv')
    return connection_gdf
def dijkstra_connection(points, point1, point2,
                        c_grid_points, line_bc,resolution):

    points.reset_index(inplace=True,drop=True)
    
    #dist_3d_matrix = distance_3d(points, points, 'X', 'Y', 'Elevation')

    

        #edges_matrix = cost_matrix(points, dist_2d_matrix, line_bc,resolution) #here it should be 
        #length_limit = resolution * 1.5
        #edges_matrix[dist_2d_matrix > math.ceil(length_limit)] = 0

        #  reduces the weights of edges already present in the cluster grid - TO BE IMPLEMENTED
        #for i in c_grid_points:
        ##    if i[0] in edges_matrix.index.values and \
        #            i[1] in edges_matrix.index.values:
        #        edges_matrix.loc[i[0], i[1]] = 0.001
        #        edges_matrix.loc[i[1], i[0]] = 0.001

        #edges_matrix_sparse = sparse.csr_matrix(edges_matrix)
    edges_matrix_sparse = sparse_dist_matrix(points, resolution*1.5)
    graph = nx.from_scipy_sparse_matrix(edges_matrix_sparse)
    for i in graph.edges:
        graph[i[0]][i[1]]['weight'] *= (points.loc[i[0],'Weight']+points.loc[i[0],'Weight'])/2
    source = points.loc[points['ID'] == int(point1['ID']), :]
    source = int(source.index.values)
    target = points.loc[points['ID'] == int(point2['ID']), :]
    target = int(target.index.values)

    # if nx.is_connected(graph): this doesn't work for Zambezia because of islands.
    #     path = nx.dijkstra_path(graph, source, target, weight='weight')
    # else:
    #     connection = pd.DataFrame()
    #     connection_cost = 999999
    #     connection_length = 999999
    #     return connection, connection_cost, connection_length, pts
    try:
        path = nx.dijkstra_path(graph, source, target, weight='weight')
        steps = len(path)
        new_path = []
        for i in range(0, steps - 1):
            new_path.append(path[i + 1])
    
        path = list(zip(path, new_path))
    
        # Creating the shapefile
        linestrings=[]
        connection_cost=0
        
        for i in path:
            point1 = points.loc[i[0],'geometry']
            point2 = points.loc[i[1],'geometry']
            connection_cost+=(points.loc[i[0],'Weight']+points.loc[i[1],'Weight'])/2*point1.distance(point2)
            linestrings.append(LineString([point1,point2]))
        connection = MultiLineString(linestrings)
        connection_length = connection.length
        comment=''
    except:
        connection=LineString([point1.geometry.values[0],point2.geometry.values[0]])
        connection_length = connection.length
        connection_cost = 99999
        comment = 'Probably PS outside of Zambezia'
    connection_gdf = gpd.GeoDataFrame({'Cost':[connection_cost],'Length':[connection_length],'Comment':[comment]},geometry=[connection],crs=32737)

    return connection_gdf

def distance_2d(df1, df2, x, y):
    """
    Find the 2D distance matrix between two datasets of points.
    :param df1: first point dataframe
    :param df2: second point dataframe
    :param x: column representing the x coordinates (longitude)
    :param y: column representing the y coordinates (latitude)
    :return value: 2D Distance matrix between df1 and df2
    """

    d1_coordinates = {'x': df1[x], 'y': df1[y]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']


    d2_coordinates = {'x': df2[x], 'y': df2[y]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = distance_matrix(df1_loc, df2_loc)
    return value
def cost_matrix(gdf, dist_3d_matrix, line_bc,resolution):
    """
    Creates the cost matrix in €/km by finding the average weight between
    two points and then multiplying by the distance and the line base cost.
    :param gdf: Geodataframe being analyzed
    :param dist_3d_matrix: 3D distance matrix of all points [meters]
    :param line_bc: line base cost for line deployment [€/km]
    :return value: Cost matrix of all the points present in the gdf [€]
    """
    # Altitude distance in meters
    weight = gdf['Weight'].values
    n = gdf['X'].size

    weight_columns = np.repeat(weight[:, np.newaxis], n, 1)
    weight_rows = np.repeat(weight[np.newaxis, :], n, 0)

    total_weight = (weight_columns + weight_rows) / 2

    # 3D distance
    value = (dist_3d_matrix * total_weight) * line_bc / 1000

    return value
def distance_3d(df1, df2, x, y, z):
    """
    Find the 3D euclidean distance matrix between two datasets of points.
    :param df1: first point dataframe
    :param df2: second point dataframe
    :param x: column representing the x coordinates (longitude)
    :param y: column representing the y coordinates (latitude)
    :param z: column representing the z coordinates (Elevation)
    :return value: 3D Distance matrix between df1 and df2
    """

    d1_coordinates = {'x': df1[x], 'y': df1[y], 'z': df1[z]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']

    d2_coordinates = {'x': df2[x], 'y': df2[y], 'z': df2[z]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = pd.DataFrame(cdist(df1_loc.values, df2_loc.values, 'euclidean'),
                         index=df1_loc.index, columns=df2_loc.index)
    return value

def sparse_dist_matrix(point_gdf, radius):
    '''Much more efficient method with sparse matrix, logarithmic rise'''
    n = len(point_gdf)
    loc = [[point_gdf.loc[i, 'geometry'].xy[0][0], point_gdf.loc[i, 'geometry'].xy[1][0]] for i in range(n)]
    # dist_matrix2 = scipy.spatial.distance_matrix(loc,loc)
    A = KDTree(loc)
    end = time.time()
    # list all pairs within 0.05 of each other in 2-norm
    # format: (i, j, v) - i, j are indices, v is distance
    D = A.sparse_distance_matrix(A, radius, p=2.0, output_type='ndarray')

    # only keep upper triangle
    DU = D[D['i'] < D['j']]

    # make sparse matrix
    result = sparse.coo_matrix((DU['v'], (DU['i'], DU['j'])), (n, n))
    result_csr = result.tocsr()
    #sparse_dist_matrix = [((i, j), result_csr[i, j]) for i, j in zip(*result_csr.nonzero())]
    
    return result_csr

def calculate_mg(clusters_list,gisele_folder,case_study,crs,mg_types):
    case_folder = gisele_folder + '/Case studies/' + case_study
    data_folder = case_folder + '/Intermediate/Optimization/all_data'
    Nodes = pd.read_csv(data_folder + '/All_Nodes.csv')
    n_clusters = int(Nodes['Cluster'].max())
    clusters_list = [*range(1,n_clusters+1)]
    cluster_powers = [Nodes.loc[Nodes['Cluster'] == i, 'MV_Power'].sum() for i in range(1,n_clusters+1)]
    cluster_population = [Nodes.loc[Nodes['Cluster'] == i, 'Population'].sum() for i in range(1,n_clusters+1)]
    clusters_list=pd.DataFrame({'Cluster':clusters_list,'Population': cluster_population,'Load [kW]': cluster_powers})
    clusters_list.to_csv(case_folder+'/Output/clusters_list.csv')
    input_profile = pd.read_csv(gisele_folder+'/general_input/Load Profile.csv').round(4)
    config = pd.read_csv(case_folder+'/Input/Configuration.csv',index_col='Parameter')
    wt=config.loc['wt','Value']
    grid_lifetime = int(config.loc['grid_lifetime','Value'])
    Nodes_gdf = gpd.GeoDataFrame(Nodes, geometry=gpd.points_from_xy(Nodes.X, Nodes.Y),
                              crs=crs)

    yearly_profile, years, total_energy = load(clusters_list,
                                               grid_lifetime,
                                               input_profile, gisele_folder, case_study)
    mg = sizing(yearly_profile, clusters_list, Nodes_gdf, wt,mg_types,gisele_folder,case_study)

    mg.to_csv(case_folder+'/Output/Microgrid.csv')

def load(clusters_list, grid_lifetime, load_profiles ,gisele_folder, case_study):
    """
    Reads the input daily load profile from the input csv. Reads the number of
    years of the project and the demand growth from the data.dat file of
    Micehele. Then it multiplies the load profile by the Clusters' peak load
    and append values to create yearly profile composed of 12 representative
    days.
    :param grid_lifetime: Number of years the grid will operate
    :param clusters_list: List of clusters ID numbers
    :return load_profile: Cluster load profile for the whole period
    :return years: Number of years the microgrid will operate
    :return total_energy: Energy provided by the grid in its lifetime [kWh]
    """
    #l()
    print("5. Microgrid Sizing")
    #l()
    case_folder = gisele_folder + '/Case studies/' + case_study

    data_michele = pd.read_table(gisele_folder+"/gisele3/michele/Inputs/data.dat", sep="=",
                                 header=None)
    print("Creating load profile for each cluster..")
    daily_profile = pd.DataFrame(index=range(1, 25),
                                 columns=clusters_list['cluster_ID'])
    for column in daily_profile:
        daily_profile.loc[:, column] = \
            (load_profiles.loc[:, str(column)])
             #* float(clusters_list.loc[clusters_list['cluster_ID']==column, 'Power [kW]'])).values
    rep_days = int(data_michele.loc[0, 1].split(';')[0])
    grid_energy = daily_profile.append([daily_profile] * 364,
                                       ignore_index=True)
    #  append 11 times since we are using 12 representative days in a year
    load_profile = daily_profile.append([daily_profile] * (rep_days - 1),
                                        ignore_index=True)

    years = int(data_michele.loc[1, 1].split(';')[0])
    demand_growth = float(data_michele.loc[87, 1].split(';')[0])
    daily_profile_new = daily_profile
    #  appending for all the years considering demand growth
    for i in range(grid_lifetime - 1):
        daily_profile_new = daily_profile_new.multiply(1 + demand_growth)
        if i < (years - 1):
            load_profile = load_profile.append([daily_profile_new] * rep_days,
                                               ignore_index=True)
        grid_energy = grid_energy.append([daily_profile_new] * 365,
                                         ignore_index=True)
    total_energy = pd.DataFrame(index=clusters_list['cluster_ID'],
                                columns=['Energy'])
    #for cluster in clusters_list['cluster_ID']:
    #    total_energy.loc[cluster, 'Energy'] = \
    #        grid_energy.loc[:, cluster].sum().round(2)
    print("Load profile created")
    #total_energy.to_csv(case_folder +'/Intermediate/Microgrid/Grid_energy.csv')
    return load_profile, years, total_energy

def sizing(load_profile, clusters_list, wt, mg_types, gisele_folder,case_study):
    """
    Imports the solar and wind production from the RenewablesNinja api and then
    Runs the optimization algorithm MicHEle to find the best microgrid
    configuration for each Cluster.
    :param load_profile: Load profile of all clusters during all years
    :param clusters_list: List of clusters ID numbers
    :param geo_df_clustered: Point geodataframe with Cluster identification
    :param wt: Wind turbine model used for computing the wind velocity
    :param mg_types: number of times to evaluate microgrids in each cluster.
                renewables fraction in michele changes accordingly
    :return mg: Dataframe containing the information of the Clusters' microgrid
    """
    speed_up = False
    case_folder = gisele_folder + '/Case studies/' + case_study
    clusters_4326 = clusters_list.to_crs('EPSG:4326')
    mg = {}
    # mg = pd.DataFrame(index=clusters_list.index,
    #                   columns=['Cluster','PV [kW]', 'Wind [kW]', 'Hydro [kW]'
    #                            'Diesel [kW]',
    #                            'BESS [kWh]', 'Inverter [kW]',
    #                            'Investment Cost [kEUR]', 'OM Cost [kEUR]',
    #                            'Replace Cost [kEUR]', 'Total Cost [kEUR]',
    #                            'Energy Demand [MWh]', 'Energy Produced [MWh]',
    #                            'LCOE [EUR/kWh]','CO2 [kg]', 'Unavailability [MWh/y]'],
    #                   dtype=float)

    for i in range(mg_types):
        mg[i] = pd.DataFrame(index=clusters_list.index,
                          columns=['Cluster','Renewable fraction index', 'PV [kW]', 'Wind [kW]', 'Diesel [kW]',
                                   'BESS [kWh]', 'Inverter [kW]',
                                   'Investment Cost [kEUR]', 'OM Cost [kEUR]',
                                   'Replace Cost [kEUR]', 'Total Cost [kEUR]',
                                   'Energy Demand [MWh]', 'Energy Produced [MWh]',
                                   'LCOE [EUR/kWh]','CO2 [kg]', 'Unavailability [MWh/y]'],
                          dtype=float)

    #save useful values from michele input data
    with open(gisele_folder+'/gisele3/michele/Inputs/data.json') as f:
        input_michele = json.load(f)
    proj_lifetime = input_michele['num_years']
    num_typ_days = input_michele['num_days']
    clusters = clusters_list['cluster_ID']
    clusters_list.reset_index(inplace=True,drop=True)
    for index in range(len(clusters)):
        # try:
        cluster_n = clusters.loc[index]
        #l()
        print('Creating the optimal Microgrid for Cluster ' + str(cluster_n))
        #l()
        load_profile_cluster = load_profile.loc[:, cluster_n]
        
        lat =clusters_4326.loc[clusters_4326['cluster_ID']==cluster_n,'geometry'].values[0].centroid.xy[1][0]
        lon = clusters_4326.loc[clusters_4326['cluster_ID']==cluster_n,'geometry'].values[0].centroid.xy[0][0]
        all_angles = pd.read_csv(gisele_folder+'/general_input/TiltAngles.csv')
        tilt_angle = abs(all_angles.loc[abs(all_angles['lat'] - lat).idxmin(),
                                      'opt_tilt'])
        if (speed_up==True and index%10==0) or speed_up==False:
            pv_prod = import_pv_data(lat, lon, tilt_angle)
            wt_prod = import_wind_data(lat, lon, wt)
        utc = pv_prod.local_time[0]
        if type(utc) is pd.Timestamp:
          time_shift = utc.hour
        else:
          utc = iso8601.parse_date(utc)
          time_shift = int(utc.tzinfo.tzname(utc).split(':')[0])
        div_round = 8760 // (num_typ_days * 24)
        length = num_typ_days * 24
        new_length= length *div_round
        # pv_avg = pv_prod.groupby([pv_prod.index.month,
        #                           pv_prod.index.hour]).mean()

        pv_avg_new=np.zeros(24*num_typ_days)
        pv_avg = pv_prod.values[0:new_length,1].reshape(24,div_round*num_typ_days,order='F')
        wt_avg_new = np.zeros(24 * num_typ_days)
        wt_avg = wt_prod.values[0:new_length].reshape(24,
                                                       div_round * num_typ_days,
                                                       order='F')
        for i in range(num_typ_days):
          pv_avg_new[i*24:(i+1)*24] = pv_avg[:,div_round*i:div_round*(i+1)].mean(axis=1)

          wt_avg_new[i * 24:(i + 1) * 24] = wt_avg[:, div_round * i:div_round * (
                  i + 1)].mean(axis=1)



        pv_avg = pd.DataFrame(pv_avg_new)
        pv_avg = pv_avg.append([pv_avg] * (proj_lifetime - 1), ignore_index=True)
        pv_avg.reset_index(drop=True, inplace=True)
        pv_avg = shift_timezone(pv_avg, time_shift)


        # wt_prod = import_wind_data(lat, lon, wt)
        # wt_avg = wt_prod.groupby([wt_prod.index.month,
        #                           wt_prod.index.hour]).mean()
        wt_avg = pd.DataFrame(wt_avg_new)
        wt_avg = wt_avg.append([wt_avg] * (proj_lifetime - 1), ignore_index=True)
        wt_avg.reset_index(drop=True, inplace=True)
        wt_avg = shift_timezone(wt_avg, time_shift)

        #todo ->implement hydro resource, for the moment creation of a fake input
        ht_avg = wt_avg

        results = start(load_profile_cluster, pv_avg, wt_avg,input_michele, ht_avg, mg_types)

        for i in range(mg_types):
          mg[i].loc[index, 'Cluster'] = str(cluster_n)
          mg[i].loc[index, 'Renewable fraction index'] = str(i)
          mg[i].loc[index, 'PV [kW]'] = results[str(i)]['inst_pv']
          mg[i].loc[index, 'Wind [kW]'] = results[str(i)]['inst_wind']
          mg[i].loc[index, 'Diesel [kW]'] = results[str(i)]['inst_dg']
          mg[i].loc[index, 'BESS [kWh]'] = results[str(i)]['inst_bess']
          mg[i].loc[index, 'Inverter [kW]'] = results[str(i)]['inst_inv']
          mg[i].loc[index, 'Investment Cost [kEUR]'] = results[str(i)]['init_cost']
          mg[i].loc[index, 'OM Cost [kEUR]'] = results[str(i)]['om_cost']
          mg[i].loc[index, 'Replace Cost [kEUR]'] = results[str(i)]['rep_cost']
          mg[i].loc[index, 'Total Cost [kEUR]'] = results[str(i)]['npc']
          mg[i].loc[index, 'Energy Produced [MWh]'] = results[str(i)]['gen_energy']
          mg[i].loc[index, 'Energy Demand [MWh]'] = results[str(i)]['load_energy']
          mg[i].loc[index, 'LCOE [EUR/kWh]'] = results[str(i)]['npc'] / \
                                                   results[str(i)]['gen_energy']
          mg[i].loc[index, 'CO2 [kg]'] = results[str(i)]['emissions']
          mg[i].loc[index, 'Unavailability [MWh/y]'] = results[str(i)]['tot_unav']
          print(mg)
        # except:
        #   print('Region too large to compute the optimal microgrid.')

    microgrid = pd.DataFrame()
    for i in range(mg_types):
        microgrid = microgrid.append(mg[i].round(decimals=4))
    #microgrid.to_csv(case_folder+'/Intermediate/Microgrid/microgrids.csv', index=False)

    return microgrid

def import_pv_data(lat, lon, tilt_angle):

    token = '556d9ea27f35f2e26ac9ce1552a3f702e35a8596  '
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()
    # Send token header with each request
    s.headers = {'Authorization': 'Token ' + token}

    url = api_base + 'data/pv'

    args = {
        'lat': lat,
        'lon': lon,
        'date_from': '2019-01-01',
        'date_to': '2019-12-31',
        'dataset': 'merra2',
        'local_time': True,
        'capacity': 1.0,
        'system_loss': 0,
        'tracking': 0,
        'tilt': tilt_angle,
        'azim': 180,
        'format': 'json',
    }
    while True:
        try:
            r = s.get(url, params=args)
            parsed_response = json.loads(r.text)
            break
        except:
            print('Problem with importing PV data. Software is in sleep mode for 30 seconds.')
            time.sleep(30)
    data = pd.read_json(json.dumps(parsed_response['data']),
                        orient='index')
    pv_prod = data
    print("Solar Data imported")

    return pv_prod


def import_wind_data(lat, lon, wt):

    token = 'c511d32b578b4ec19c3d43c1a3fffb4cad5dc4d2'
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()
    # Send token header with each request
    s.headers = {'Authorization': 'Token ' + token}
    url = api_base + 'data/wind'
    args = {
        'lat': lat,
        'lon': lon,
        'date_from': '2019-01-01',
        'date_to': '2019-12-31',
        'capacity': 1.0,
        'height': 50,
        'turbine': str(wt),
        'format': 'json',
    }

    # Parse JSON to get a pandas.DataFrame
    r = s.get(url, params=args)
    parsed_response = json.loads(r.text)

    data = pd.read_json(json.dumps(parsed_response['data']),
                        orient='index')
    wt_prod = data
    print("Wind Data imported")

    return wt_prod
def shift_timezone(df, shift):
    """
    Move the values of a dataframe with DateTimeIndex to another UTC zone,
    adding or removing hours.
    :param df: Dataframe to be analyzed
    :param shift: Amount of hours to be shifted
    :return df: Input dataframe with values shifted in time
    """
    if shift > 0:
        add_hours = df.tail(shift)
        df = pd.concat([add_hours, df], ignore_index=True)
        df.drop(df.tail(shift).index, inplace=True)
    elif shift < 0:
        remove_hours = df.head(abs(shift))
        df = pd.concat([df, remove_hours], ignore_index=True)
        df.drop(df.head(abs(shift)).index, inplace=True)
    return df

def calculate_power_on_lines(lines,clusterss,substations):
    'This function should calculate the power flow on each line'
    def find_terminal_nodes(lines):
        lines_nodes = lines['ID1'].to_list() + lines['ID2'].to_list()
        count = Counter(lines_nodes)
        terminal_nodes = [i for i in count if (count[i]==1) & (i not in substation_nodes)]
        return terminal_nodes
    substation_nodes = substations.ID.to_list()
    
    power_in_nodes = dict.fromkeys(clusterss['ID'], 0)
    for node in power_in_nodes:
            power_in_nodes[node] = clusterss.loc[clusterss['ID']==node,'Power [kW]'].values[0]
            
    lines.reset_index(inplace=True,drop=True)
    lines['ID1']= [int(i.split('-')[0])for i in list(lines['name'].values)]
    lines['ID2'] = [int(i.split('-')[1])for i in list(lines['name'].values)]
    lines_copy = lines.copy()
    terminal_nodes = find_terminal_nodes(lines)
    
    
    
    while True:
        select_node=terminal_nodes[0]
        ind = lines_copy.index[(lines_copy['ID1']==select_node) | (lines_copy['ID2']==select_node)][0]
        lines.loc[ind,'Power [kW]'] = power_in_nodes[select_node]
        next_node = [i for i in list(lines_copy.loc[ind,['ID1','ID2']].values) if i!=select_node][0]
        if next_node in clusterss['ID'].to_list():
            power_in_nodes[next_node]+= power_in_nodes[select_node] 
        
        lines_copy.drop(ind,inplace=True)
        if lines_copy.empty:
            break
        terminal_nodes = find_terminal_nodes(lines_copy)
        return lines
   