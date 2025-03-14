# -*- coding: utf-8 -*-
"""
@author: Iurii Savvateev
"""

import pandas as pd

import numpy as np

import pickle

import os

directory=r'D:\Parcellation\Github\Cluster_perturbation_analysis'
templatepath=os.path.join(directory,'Template') 
statistics_path=os.path.join(directory,'Cluster_statistics')
os.chdir(directory)

with open('zmaps_sorted_per_cluster.pkl', 'rb') as f:
    zmaps = pickle.load(f)


zmap_type_condition=['GFP_masked','Gi_masked']

regions=pd.read_excel(os.path.join(templatepath,"ROI_List_Allen_213_to_165_original.xlsx"),engine='openpyxl')
# Remove areas that are not used in the computation of the correlation matrix: Hin/Midbrains and OLF
for a in ['OLF','Hindbrain','Midbrain']:
    regions.drop(regions.loc[regions['Area']==a].index,inplace=True)

# Getting the list of unique regions for further analysis
regions_unique=regions['Area'].unique() 

#construct the dataframes for each condition and cluster containing zscores for all regions (mean within the region)
for zmap_type in zmap_type_condition: # loop through the conditions
    for cluster in range(1, 5):  # loop through the clusters
        animal_list=list(zmaps[zmap_type][f'cluster_{cluster}'].keys())
        #writer=pd.ExcelWriter(os.path.join(zmaps_melodic_path,f'{zmap_type}_pre_melodics cluster_{cluster}.xlsx'))
        data_cluster_combined=pd.DataFrame(columns=['Area','Abbreviation','Full name'])
        for animal in animal_list: #combine data from different animals
            
             data_cluster_individual_df=pd.DataFrame(columns=['Area','Abbreviation','Full name',\
                                                              f'{animal}']) 
             
             for single_region in regions_unique: # combine data from different regions
                 for ind in regions.loc[regions['Area'] == single_region, 'Abbreviation']:
                
                    full_name = regions.loc[(regions['Area'] == single_region) &
                                        (regions['Abbreviation'] == ind), 'Full name'].values
                    zscore = zmaps[zmap_type][f'cluster_{cluster}'][animal][single_region]\
                    [ind]['zscores']
                    
                    data_individual_row=pd.DataFrame.from_dict({data_cluster_individual_df.columns[0]:single_region,\
                      data_cluster_individual_df.columns[1]:ind,\
                      data_cluster_individual_df.columns[2]:full_name,\
                           
                      data_cluster_individual_df.columns[3]:np.mean(zscore)})
                        
                    
                    data_cluster_individual_df=pd.merge(data_cluster_individual_df,data_individual_row,\
                                                        how='outer')
             data_cluster_combined=pd.merge(data_cluster_combined,data_cluster_individual_df, how='outer')
         
        data_cluster_combined.to_excel(os.path.join(statistics_path,\
                                                    f'{zmap_type}_cluster_{cluster}.xlsx'))
        del data_cluster_combined


        
