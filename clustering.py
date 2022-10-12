from gisele3.functions import *
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import *
import rasterio
import pandas as pd
from sklearn.cluster import DBSCAN
import time


operation = 'Clustering' # 'Clustering' or 'Delaunay'
crs = 32736
#Population = gpd.read_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\grid_points')
#Population.drop(['X','Y','LandCover','ID'],axis=1,inplace=True)
#Population_filtered = Population[Population['Population']>0]
Population_gdf = gpd.read_file(r'D:\OneDrive - Politecnico di Milano\EPSlab_Com2\gisele_v03\Data\Zambezia\Population_Zambezia')
Population_gdf= Population_gdf.to_crs('EPSG:'+str(crs))
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