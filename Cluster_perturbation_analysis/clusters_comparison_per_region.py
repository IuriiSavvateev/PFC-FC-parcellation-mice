# -*- coding: utf-8 -*-
"""
@author: Iurii Savvvateev
"""

import pandas as pd
import numpy as np

import pickle

import os
import scipy.io
from scipy import stats



# Defining working directory and path

directory=r'D:\Parcellation\Github\Cluster_perturbation_analysis'
templatepath=os.path.join(directory,'Template') 
os.chdir(directory)

# Loading previously stored dictionary containing zstats per region, per cluster
# Calculated with zmaps_calculation.py 

with open('zmaps_sorted_per_condition.pkl', 'rb') as f:
    zmaps = pickle.load(f)


def stat_comparison_one_sample(x,value):
    '''Function performing statistical comparison of a distribution "x" to a
    known value. By default an alternative hypothesis is used
    '''
    if len(x)<3:
        #test_name=float("NaN")
        s=float("NaN")
        p=float("NaN")
    else:
        s_x_norm,p_x_norm=stats.shapiro(x) # Checking for the normality of the distribution x
        
        if p_x_norm >0.05:
            s,p=stats.ttest_1samp(x,value,alternative='less')
            #test_name="t-test"  #t-test
        else: 
            s,p=scipy.stats.wilcoxon(x-value,alternative='less')
            #check whether the values are higher than the threshold
            #test_name="Wilcoxon (diff)" #Wilcoxon
            # Welch correction of independent t-test     
    return p 


def stat_comparison(x,y):
    '''Function performing statistical comparison of two distributions
    '''
    if (len(x)<3) | (len(y)<3):
        test_name=float("NaN")
        s=float("NaN")
        p=float("NaN")
    else:
        s_hom,p_hom=stats.levene(x,y) # Homogeneity of varience check
        s_x_norm,p_x_norm=stats.shapiro(x) # Checking for the normality of the distribution x
        s_y_norm,p_y_norm=stats.shapiro(y) # Checking for the normality of the distribution y
        
        if (p_hom >0.05) & (p_x_norm >0.05) & (p_y_norm >0.05):
            s,p=stats.ttest_ind(x,y)
            test_name="t-test"  #t-test
        elif (p_hom <0.05) & (p_x_norm >0.05) & (p_y_norm >0.05): 
            s,p=stats.ttest_ind(x,y,equal_var=False) 
            test_name="Welch-ttest"  # Welch correction of independent t-test
        else: 
            s,p=scipy.stats.mannwhitneyu(x,y)
            test_name="U-test" #U-test
            # Welch correction of independent t-test
       
    return test_name,s,p 



zmap_type_condition=['GFP_masked','Gi_masked'] 
regions=pd.read_excel(os.path.join(templatepath,"ROI_List_Allen_213_to_165_original.xlsx"),engine='openpyxl')

# Remove areas that are not used in the computation of the correlation matrix: Hin/Midbrains and OLF
for a in ['OLF','Hindbrain','Midbrain']:
    regions.drop(regions.loc[regions['Area']==a].index,inplace=True)

# Getting the list of unique regions for further analysis
regions_unique=regions['Area'].unique() 


compared_conditions=[n for n in zmap_type_condition]
comparison_path=os.path.join(directory,'Zmaps_comparison')

condition0=zmap_type_condition[0]
condition1=zmap_type_condition[1]
z_threshold=1.96


# Comparison is conducted across the conditions for each of the cluster
data_comparison=['Area','Abbreviation','Full name',\
                         
                         f'{compared_conditions[0]}_mean',\
                         f'{compared_conditions[0]}_std',\
                         f'{compared_conditions[0]}_ss_{z_threshold}',\
                         f'{compared_conditions[0]}_size',\
                             
                         f'{compared_conditions[1]}_mean',\
                         f'{compared_conditions[1]}_std',\
                         f'{compared_conditions[1]}_ss_{z_threshold}',\
                         f'{compared_conditions[1]}_size',\

                         'comparison_test','test statistics','p-value'\
                         ]



for cluster_number in range(1,5): # for 4 cluster solution
#create a separate dataframe for each cluster
        data_cluster_comparison_df=pd.DataFrame(columns=data_comparison)
        for single_region in regions_unique:
            for ind in regions.loc[regions['Area']== single_region,'Abbreviation']:

                
                full_name=regions.loc[(regions['Area']== single_region) & \
                                      (regions['Abbreviation']== ind),'Full name'].values
                    
                condition_0_dict=zmaps[condition0][single_region][ind][f'cluster_{cluster_number}']
                condition_1_dict=zmaps[condition1][single_region][ind][f'cluster_{cluster_number}']
                
                
                # Calculation of the mean across animals
    
                    
                # Take z-scores for each animal and calculate the mean value across the region
                
                condition_0_array=np.mean([condition_0_dict[animal_0]['zscores'] \
                            for animal_0 in list(condition_0_dict.keys())],axis=1)
                
                
                condition_1_array=np.mean([condition_1_dict[animal_1]['zscores'] \
                            for animal_1 in list(condition_1_dict.keys())],axis=1)
                        
                # checking with one sample test the difference for each animal
               
                    
                test_name,s_value,p_value=stat_comparison(condition_0_array, condition_1_array)
                
                p_value_0=stat_comparison_one_sample(abs(np.array(condition_0_array)),z_threshold) 
                p_value_1=stat_comparison_one_sample(abs(np.array(condition_1_array)),z_threshold)
                
            
                                         
                data_row={data_cluster_comparison_df.columns[0]:single_region,\
                              data_cluster_comparison_df.columns[1]:ind,\
                              data_cluster_comparison_df.columns[2]:full_name,\
                                  
                              
                              data_cluster_comparison_df.columns[3]:np.mean(condition_0_array),\
                              data_cluster_comparison_df.columns[4]:np.std(condition_0_array),\
                              data_cluster_comparison_df.columns[5]:p_value_0,\
                              data_cluster_comparison_df.columns[6]:len(condition_0_array),\
                              
                              data_cluster_comparison_df.columns[7]:np.mean(condition_1_array),\
                              data_cluster_comparison_df.columns[8]:np.std(condition_1_array),\
                              data_cluster_comparison_df.columns[9]:p_value_1,\
                              data_cluster_comparison_df.columns[10]:len(condition_1_array),\
                                                                                           
                                                                                           
                              data_cluster_comparison_df.columns[11]:test_name,\
                              data_cluster_comparison_df.columns[12]:s_value,\
                              data_cluster_comparison_df.columns[13]:p_value}   
                        
                data_cluster_comparison_df=data_cluster_comparison_df.append(data_row,ignore_index=True)
        
                data_cluster_comparison_df.loc[data_cluster_comparison_df['p-value']<=0.05, \
                                                                      "Statistical Significance"]=True        
                data_cluster_comparison_df.loc[data_cluster_comparison_df['p-value']>0.05, \
                                                                      "Statistical Significance"]=False        
                    
        saved_name=f'comparison-cluster-{cluster_number}-{compared_conditions[0]}-{compared_conditions[1]}.xlsx'
    
        data_cluster_comparison_df.to_excel(os.path.join(comparison_path,saved_name),index=False)
        del  data_cluster_comparison_df


