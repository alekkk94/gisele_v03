from gisele3.functions import *
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import *
import rasterio
import pandas as pd
from sklearn.cluster import DBSCAN
import math
import time
crs = 32736
regions = gpd.read_file(os.path.join(r'C:\Users\alekd\Politecnico di Milano\Documentale DENG - Mozambique\Admin','gadm36_MOZ_1.shp'))
regions_crs=regions.to_crs(crs)
zambezia_polygon = regions.loc[regions['NAME_1']=='Zambezia','geometry'].values[0][45] #the mainland
zambezia_polygon_crs = regions_crs.loc[regions_crs['NAME_1']=='Zambezia','geometry'].values[0][45]

#calculate a fake number of substations needed based on the radius of 3km per substation + a bit of slack.
#basically, considering the inner square of a circle with a radius 3km, we can find that the surface is (3km/sqrt(2))^2

overall_area = zambezia_polygon_crs.area/1000000 #km^2

area_of_substation = (3/math.sqrt(2))**2 #in km^2 considering a radius of 3km
coef_routing = 0.3 # do something better to account for the fact that we if there are people in the entire circle, the area supplied will be much less
# then the full surface. Basically, try to consider like a weight based on the amount of points inside that rectangle.
number_substations_based_on_area = int(overall_area/area_of_substation/coef_routing)

operation = 'Clustering' # 'Clustering' or 'Delaunay'

Population_gdf = gpd.read_file(r'C:\Users\alekd\OneDrive - Politecnico di Milano\PhD studies\Papers\gisele-3 paper\Data\Zambezia\Population_Zambezia')
Population_gdf= Population_gdf.to_crs('EPSG:'+str(crs))
load_capita = 20 #Watts
overall_load = Population_gdf.Population.sum()*load_capita/1000000

number_substations_based_on_power = overall_load/0.4
location_folder = r'C:\Users\alekd\OneDrive - Politecnico di Milano\PhD studies\Papers\gisele-3 paper\Data\Zambezia'
Population_gdf, centroids = run_k_means(Population_gdf,int(number_substations_based_on_power),location_folder,crs)



if operation == 'Clustering':

    clustering_option = 'DBSCAN'
    action = 'Cycle_clustering' # 'Sensitivity' 'cycle clustering'
    start=time.time()
    if clustering_option == 'DBSCAN':
        if action=='Sensitivity':
            sensitivity_output_no_clusters=pd.DataFrame()
            sensitivity_output_perc_clustered = pd.DataFrame()
            for min_eps in [283]:
                for min_pop in range(720,811,30):
                    Population_clustered, n_clusters, percentage_clustered = run_DBSCAN(Population_gdf, min_eps, min_pop)
                    sensitivity_output_no_clusters.loc[min_eps,min_pop] = n_clusters
                    sensitivity_output_perc_clustered.loc[min_eps,min_pop] = percentage_clustered
                    print('Executed DBSCAN clustering for min_eps='+ str(min_eps)+' and min_pop='+str(min_pop)+'.')
        elif action == 'Cycle_clustering':
            min_eps=283
            for min_pop in [800,280,175,125,90,65,50,35,20]:
                Population_clustered, n_clusters, percentage_clustered = run_DBSCAN(Population_gdf, min_eps, min_pop)
                Population_clustered_filtered = Population_clustered[Population_clustered['Clusters'] != -1]
                Population_clustered_filtered.to_file(
                    r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia_clustered_' + str(
                        min_pop))
        else:
            min_eps = 283
            min_pop = 800
            Population_clustered,n_clusters,percentage_clustered = run_DBSCAN(Population_gdf,min_eps,min_pop)
        print('There are '+ str(n_clusters)+ ' clusters in the area, with '+str(percentage_clustered)+ '% of the people being clustered.')
    elif clustering_option=='OPTICS':
        Population_clustered, n_clusters, percentage_clustered = run_OPTICS(Population_gdf, min_eps, min_pop)
    end=time.time()
    print('Time required is '+str(end-start)+' seconds.')
    if sensitivity_analysis:
        sensitivity_output_no_clusters.to_csv(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\sensitivity_clusters3.csv')
        sensitivity_output_perc_clustered.to_csv(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\sensitivity_perc_clusters3.csv')
    else:
        print(n_clusters)
        Population_clustered['ID']=Population_clustered.index
        #Population_clustered.to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia_clustered')
        #Population_clustered.to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia_clustered.gpkg',layer='pop',driver='GPKG')
        Population_clustered_filtered = Population_clustered[Population_clustered['Clusters']!=-1]
        Population_clustered_filtered.to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia_clustered_'+str(min_pop))

elif operation == 'Delaunay':
    print('Starting the delaunay triangulation stuff...')
    dist_limit = 2 #km
    graph,new_lines = delaunay_test(Population_gdf,crs)
    nx.write_gpickle(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\delaunay_graph.pkl')
    new_lines.to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\delaunay')
    new_lines[new_lines['length']<dist_limit].to_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\delaunay1')

#Gi seces liniite so progresivno namaluvanje na dolzinata. Kako se formiraat clasteri, taka pravis presmetki za troshoci za generacisko portfolio i substations.


# Clustering analysis in June 2023
from functions import *
point_gdf = point_density(point_gdf,calculate_percentiles=False,radius=568)
point_gdf = point_density(point_gdf,calculate_percentiles=False,radius=1000)
point_gdf = point_density(point_gdf,calculate_percentiles=False,radius=2000)
#point_gdf = point_density(point_gdf,calculate_percentiles=False,radius=3000)
point_gdf = point_density(point_gdf,calculate_percentiles=False,radius=142)
point_gdf.sort_values(by='Pop_142').reset_index(drop=True)['Pop_142'].plot()
point_gdf.to_file(r'C:\Users\alekd\PycharmProjects\gisele_v03\Data\Zambezia_clustering_JUNE\partial_zambezia')

#let's do some analysis considering trafo deployment. Eventually, we can have difference distance threshholds.
power_household = 0.3
contemporary_coef =0.3
max_loadibility = 0.6 # in pu
power_factor=0.9
kva_400 = 400/power_household/contemporary_coef*max_loadibility*power_factor
kva_250 = 250/power_household/contemporary_coef*max_loadibility*power_factor
kva_160 = 160/power_household/contemporary_coef*max_loadibility*power_factor
kva_100 = 100/power_household/contemporary_coef*max_loadibility*power_factor
kva_50 = 50/power_household/contemporary_coef*max_loadibility*power_factor
kva_25 = 25/power_household/contemporary_coef*max_loadibility*power_factor

overall_power = point_gdf.Population.sum()*power_household*contemporary_coef/power_factor/max_loadibility #transformers needed 
#without considering distance constraint

#Let's try a layered approach going from the most dense areas
first_layer = point_gdf.loc[point_gdf['Pop_1000']>=1500]
pop_first_layer = first_layer['Population'].sum()
second_layer =  point_gdf.loc[point_gdf['Pop_1000']>=900]
pop_second_layer =  second_layer['Population'].sum()