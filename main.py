import os
os.chdir(r'C:\Users\alekd\PycharmProjects\gisele_v03')
import geopandas as gpd
import pandas as pd
from shapely.geometry import *
import rasterio
import pandas as pd
import math
import time
import scipy
from gisele3.functions import *
import rustworkx as rx
crs = 32736
mozambique_folder = r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\_vania_2.0\Data\Case_Study\Country Level\Mozambique'
#processing the population
buildings_folder = 'Buildings'
admin_folder = r'Admin'
pop_folder= 'Population'

regions = gpd.read_file(os.path.join(mozambique_folder,admin_folder,regional_division_file))
regions_crs=regions.to_crs(crs)
zambezia_polygon = regions.loc[regions['NAME_1']=='Zambezia','geometry'].values[0][45] #the mainland
zambezia_polygon_crs = regions_crs.loc[regions_crs['NAME_1']=='Zambezia','geometry'].values[0][45] #the mainland

resolution=100
import geopandas as gpd
points = gpd.read_file(r'C:\Users\alekd\OneDrive - Politecnico di Milano\PhD studies\Papers\gisele-3 paper\Data\Zambezia\Population_kmeans')
points = points.loc[points['Cl_Kmeans'] == 0, 'geometry']
unique_clusters = len(points['Cl_Kmeans'].unique())
hulls = [polygon_around_points(points.loc[points['Cl_Kmeans']==i,'geometry']) for i in range(unique_clusters)]
hulls_gdf = gpd.GeoDataFrame({'Cl_Kmeans':[*range(unique_clusters)]},geometry = hulls,crs=crs)
hulls_gdf.to_file(r'C:\Users\alekd\OneDrive - Politecnico di Milano\PhD studies\Papers\gisele-3 paper\Data\testing\hull')

Population_gdf_clustered,n_clusters,percentage_clustered = run_OPTICS(points,100)

#try to change kmeans in a way that we can update the weights not only on the many points in the network, but also of the centroid.
start=time.time()
Population_gdf,n_clusters,percentage_clustered = run_DBSCAN(points,min_eps=2*100*math.sqrt(2),min_pop=35)
end=time.time()


loc = [[points.loc[i,'geometry'].xy[0][0],points.loc[i,'geometry'].xy[1][0]] for i in range(100000)]
a=time.time()

dist_matrix = scipy.spatial.distance.pdist(loc,metric='euclidean')
b=time.time()
x = 8 #select element
n=10 #elements
list=[]
counter = 0
a=[*reversed(range(n))]
for i in reversed(range(x)):
    list.append(dist_matrix[sum(a[0:counter])+i])
    counter+=1


dist_matrix2 = scipy.spatial.distance_matrix(loc,loc)
c=time.time()
print(c-b)


'''Much more efficient method with sparse matrix, logarithmic rise'''


#Calculate the density of point, creating a smooth distribution
points = point_density(points, radius=284)


points.to_file(r'C:\Users\alekd\OneDrive - Politecnico di Milano\PhD studies\Papers\gisele-3 paper\Data\Zambezia\Population_statistics_density')

# Let's create the steiner tree of the clusters in Zambezia -> March 2023

#Step 1. Open the clusters, estimate population find centroids and create a graph
clusters = gpd.read_file(r'C:/Users/alekd/Politecnico di Milano/Marco Merlo - @TERESA/Bibliography/VANIA/VANIA_Output_Zambezia/Clusters/Complete/clusters_above_100.shp')
clusters.to_crs('EPSG:'+str(crs),inplace=True)
clusters=clusters[['geometry']]
pop_zambezia = gpd.read_file(r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia\Population_Zambezia')
pop_zambezia.to_crs('EPSG:'+str(crs),inplace=True)

pop_zambezia_new = gpd.sjoin(pop_zambezia,clusters,how='inner',op='within')
clusters['Population'] = pop_zambezia_new.groupby('index_right').agg('sum')['Population']
clusters['Population']  = clusters['Population'].fillna(0)
substations = gpd.read_file(r'C:/Users/alekd/Politecnico di Milano/Marco Merlo - @TERESA/Bibliography/VANIA/Mozambique/@Substations/SubstationZambezia/SubstationsZambezia.shp')



substations_zambezia = gpd.read_file(r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia_2023\substations_zambezia\substations_zambezia.shp')
substations_zambezia.drop(columns=['Min','Population','ID'],inplace=True)
substations_zambezia.to_crs('EPSG:'+str(crs),inplace=True)
clusters_centroids = clusters.copy()
clusters_centroids.geometry=clusters_centroids.geometry.centroid
clusters_centroids['ID'] = [*range(len(clusters_centroids))]

substations_zambezia['ID']=[*range(len(clusters_centroids),len(clusters_centroids)+len(substations_zambezia))]
all_nodes = clusters_centroids.append(substations_zambezia)
terminal_nodes = all_nodes.ID.to_list()
graph=nx.Graph()
lines=gpd.GeoDataFrame()
graph, lines = delaunay_test(all_nodes, all_nodes.crs,return_lines=True)
G = rx.PyGraph()
indices = G.add_nodes_from(graph.nodes())
for i in graph.edges.data():
    print(i)
    G.add_edge(i[0], i[1], i[2]['weight'])

start=time.time()
T = rx.steiner_tree(G, terminal_nodes,weight_fn=float)
end=time.time()
node_indices = T.node_indices()
edge_indices = T.edge_indices()
steiner_tree_representation = pd.DataFrame(columns=['Line','weight'])
counter=0
id1 = []
id2=[]
weight=[]
for edge in edge_indices:
    steiner_tree_representation.loc[counter,'id1'] = T.get_edge_endpoints_by_index(edge)[0]
    steiner_tree_representation.loc[counter,'id2'] = T.get_edge_endpoints_by_index(edge)[1]
    steiner_tree_representation.loc[counter,'weight'] = T.get_edge_data_by_index(edge)
    counter+=1
geom = []
for i,row in steiner_tree_representation.iterrows():
    id1 = row['id1']
    id2 = row['id2']
    point1 = all_nodes.loc[all_nodes['ID']==id1,'geometry'].values[0]
    point2 = all_nodes.loc[all_nodes['ID']==id2,'geometry'].values[0]
    line = LineString([point1,point2])
    geom.append(line)
steiner_tree_representation_gdf = gpd.GeoDataFrame(steiner_tree_representation,geometry=geom,crs=clusters_centroids.crs)
steiner_tree_representation_gdf.to_file(r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia_2023\steiner_above_100')

'''This part is about finding the closest substation to each cluster - in this case, the centroid'''
# find the distances between each cluster and each substation
clusters_centroids=clusters_centroids[['ID','Population','geometry']]
dist_matrix = rx.floyd_warshall(T,weight_fn=float)
for i,row in clusters_centroids.iterrows():
    for j,row1 in substations_zambezia.iterrows():
        clusters_centroids.loc[i,row1['Name']] = round(dist_matrix[row['ID']][row1['ID']])

clusters_centroid_reduced = clusters_centroids.drop(columns=['ID','geometry','Population'])
clusters_centroid_reduced['Min'] = clusters_centroid_reduced.min(axis='columns')

for i,row in clusters_centroids.iterrows():
    closest = clusters_centroid_reduced.loc[i,'Min']
    clusters_centroids.loc[i,'Min']= row[3:][row[3:]==closest].index.values[0]

clusters_centroids.to_file(r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia_2023\centroids_with_distances')

substations_zambezia['Min'] = substations_zambezia['Name']


summed_pop = clusters_centroids.groupby('Min').agg('sum')['Population']
for i,row in substations_zambezia.iterrows():
    try:
        substations_zambezia.loc[i,'Population'] = summed_pop[row['Name']]
    except:
        substations_zambezia.loc[i,'Population'] = 0

power_per_capita = 0.1 #kW
substations_zambezia['Power'] = substations_zambezia['Population']*power_per_capita/1000 #MW
substations_zambezia.to_file(r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia_2023\substations_zambezia')

# Now let's separate each tree and calculate the voltage.
new_lines = steiner_tree_representation_gdf.copy()
for i,row in steiner_tree_representation_gdf.iterrows():
    try: 
        if not clusters_centroids.loc[clusters_centroids['ID'] ==row['id1'],'Min'].values[0] == clusters_centroids.loc[clusters_centroids['ID'] ==row['id2'],'Min'].values[0]:
            new_lines.drop(i,inplace=True)
    except: #this just means that one of the points is a substation
        pass
new_lines.to_file(r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia_2023\steiner_cut')


# Testing the limits of steiner tree in rustworkx - 98618 edges with 33243 nodes
Population_small_area = gpd.read_file(r'C:/Users/alekd/PycharmProjects/gisele_v03/Data/Zambezia/Population_small_area/Population_small_area.shp')
Population_small_area = gpd.read_file(r'C:\Users\alekd\OneDrive - Politecnico di Milano\PhD studies\Papers\gisele-3 paper\Data\Zambezia\Population_kmeans')
Population_small_area.to_crs('EPSG:'+str(crs),inplace=True)
Population_small_area['ID'] = Population_small_area.index



graph=nx.Graph()
lines=gpd.GeoDataFrame()
graph, lines = delaunay_test(Population_small_area, Population_small_area.crs,return_lines=False)
G = rx.PyGraph()
indices = G.add_nodes_from(graph.nodes())
for i in graph.edges.data():
    #print(i)
    G.add_edge(i[0], i[1], i[2]['weight'])
    
T = rx.minimum_spanning_tree(G, weight_fn= float)
T = rx.steiner_tree(G, indices,weight_fn=float)
node_indices = T.node_indices()
edge_indices = T.edge_indices()

steiner_tree_representation = pd.DataFrame(columns=['Line','weight'])
counter=0
id1 = []
id2=[]
weight=[]
for edge in edge_indices:
    id1.append(T.get_edge_endpoints_by_index(edge)[0])
    id2.append(T.get_edge_endpoints_by_index(edge)[1])
    weight.append(T.get_edge_data_by_index(edge))
    counter+=1
geom_1 = Population_small_area.loc[id1,'geometry'].to_list()
geom_2 = Population_small_area.loc[id2,'geometry'].to_list()
geom_lines = [LineString([geom_1[i],geom_2[i]]) for i in range(len(id1))]

steiner_tree_representation_gdf = gpd.GeoDataFrame({'ID1':id1,'ID2':id2,'length':weight},geometry=geom_lines,crs=crs)
steiner_tree_representation_gdf.to_file(r'C:/Users/alekd/PycharmProjects/gisele_v03/Data/Zambezia_2023/MST_Zambezia')
