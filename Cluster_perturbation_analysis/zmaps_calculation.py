# -*- coding: utf-8 -*-
"""
@author: Iurii Savvvateev
"""

import pandas as pd
import numpy as np


import nibabel

import os
import scipy.io
from scipy import stats

import pickle



directory=r'D:\Parcellation\Github\Cluster_perturbation_analysis'
os.chdir(directory)


def cluster_mask(cluster_number,mask):
    '''Function that extracts indices that correspond to a specific region
    '''
    x=np.zeros(mask.shape, dtype=int)
    z=np.argwhere(mask==cluster_number)
    for i in range(len(z)):
        x[z[i][0]][z[i][1]][z[i][2]]=1
    return (x.astype(bool))

def Nodes_213_165(mask_brain_old,nodes_good):
    """ Function to select 165 rois based on "GOOD NOTES" file 
        from 213 rois
        Inputs: mask_213 (numpy)
                nodes_good (list)
        Returns: mask_165 (numpy)
        """
    mask_labels_new=np.zeros(mask_brain_old.shape, dtype=int)

    for i in np.nditer(nodes_good): # take the good nodes only
        x=np.argwhere(mask_brain_old==i) # extract the index of all good nodes
        for j in range(len(x)):
            mask_labels_new[x[j][0]][x[j][1]][x[j][2]]=i # form a mask consisting of good nodes only
    return mask_labels_new

def individual_cluster(regions,roi,cluster_sel,path,appendix=''):
    """ Summarize information from individual clusters
        Inputs:
        regions -DataFrame with all regions
        roi - dictionary with the filtered zvalues
        cluster_sel- selected cluster
        path - where statistics should be saved in excel format
        Returns:
        DataFrame containing individual cluster statistics
        ExcelTable with individual region statistics in the specified path
    """
    data_cluster_individual=['Area','Abbreviation','Full name',\
                         f'cluster_{cluster_sel}_mean',\
                         f'cluster_{cluster_sel}_std',\
                         f'cluster_{cluster_sel}_size',\
                         f'cluster_{cluster_sel}_sem',\
                         ]
    data_cluster_individual_df=pd.DataFrame(columns=data_cluster_individual)   
    for single_region in regions_unique:
        for ind in regions.loc[regions['Area']== single_region,'Abbreviation']:
        #print(f'{single_region=},{ind=}')
            full_name=regions.loc[(regions['Area']== single_region) & \
                              (regions['Abbreviation']== ind),'Full name'].values
            roi_1=roi[single_region][ind][f'cluster_{cluster_sel}']['zscores']

            # internal check, if the size of the region is more then 3 voxels that have statistically significant values
            roi_1_len=len(roi_1)
        
            if roi_1_len >=1: # currently remove this check by setting cluster volume to 1
        
          
                data_individual_row={data_cluster_individual_df.columns[0]:single_region,\
                  data_cluster_individual_df.columns[1]:ind,\
                  data_cluster_individual_df.columns[2]:full_name,\
                      
                  
                data_cluster_individual_df.columns[3]:np.mean(roi_1),\
                data_cluster_individual_df.columns[4]:np.std(roi_1),\
                data_cluster_individual_df.columns[5]:roi_1_len,\
                data_cluster_individual_df.columns[6]:np.std(roi_1)/np.sqrt(roi_1_len),\
                 }   
            
                data_cluster_individual_df=data_cluster_individual_df.append(data_individual_row,ignore_index=True)
            else : 
                continue    

    os.chdir(path)
    data_cluster_individual_df.to_excel(f'cluster_{cluster_sel}_{appendix}.xlsx')
    return data_cluster_individual_df

def substract_cluster_from_zmap(cluster,zmap):
    """ Removes voxels assigned to the cluster from the zmap
    Inputs: cluster-3D numpy array, zmap- 3D numpy array matching cluster.shape
    Outputs: zmap_masked - 3D numpy array with the original "True" values from 
    cluster substituted by NaN
    """ 
    
    #Check if the dimensionalities of the matrices match
    if cluster.shape !=zmap.shape:
        raise TypeError("Matrices should be of the same dimensionality")
  
    # Exchange zeros and ones in the matrix
    cluster_0=np.where(cluster==0)
    cluster_1=np.where(cluster==1)
    cluster[cluster_0]=1
    cluster[cluster_1]=np.nan
    
    #multiply the cluster to the zmap with nan at the cluster values
    zmap_masked=zmap*cluster 
    
    return zmap_masked



# Selecting a mask for the target area
maskpath=os.path.join(directory, 'Brain_mask')
nodes = scipy.io.loadmat(os.path.join(maskpath,'GOOD_NODES.mat'))
nodes_good=nodes["GOOD_NODES"]

# Selecting a template with the region names
templatepath=os.path.join(directory,'Template')
mask_brain_210=nibabel.load(os.path.join(templatepath,'mask_Oh_V3_r_QBI_new_Rocchi_Template.nii.gz')).get_fdata()
mask_labels_165_1_172=Nodes_213_165(mask_brain_210,nodes_good) #Subselect only the chosen nodes listed in GOOD_NODES file

# Since labels in mask_labels_165_1_172 correspond to the order from 213 regions file (with the max value being 172)
# we substitute each value by its index, thus having labels from 1 till 165, 0 - background
# in this manner labels are in according to the Excel tabel

mask_labels_165_1_165=np.zeros(mask_labels_165_1_172.shape,dtype=int)
for ind,value in enumerate(np.unique(mask_labels_165_1_172)): # create tuple (new index(1-165), old value(1-172))
        positions=np.argwhere(mask_labels_165_1_172==value) # take all positions of old value (1-172)
        for p in positions:
            mask_labels_165_1_165[p[0]][p[1]][p[2]]=ind 
            # form a mask consisting with the clusters named from 1 till 165


regions=pd.read_excel(os.path.join(templatepath,"ROI_List_Allen_213_to_165_original.xlsx"),engine='openpyxl')

# Remove areas that are not used in the computation of the correlation matrix: Hin/Midbrains and OLF
for a in ['OLF','Hindbrain','Midbrain']:
    regions.drop(regions.loc[regions['Area']==a].index,inplace=True)

# Getting the list of unique regions for further analysis
regions_unique=regions['Area'].unique() 

zmap_type_condition=['GFP_masked','Gi_masked'] # identify to compared conditions


paired=False # Parameter to be used if the data are paired, but the finally discussed datasets 
# were from the different animals


cluster_substraction=False #include cluster voxels in the final zmap
# Parameter could be used to exclude cluster voxels from the zmap


# Sorted_per_condition
zmaps={}
for zmap_type in zmap_type_condition: 
    zmaps_path=os.path.join(directory, f'Analysis_{zmap_type}','zstats') 
    zmaps[zmap_type]={} # creating a dictionary for further storage
    for single_region in regions_unique:
        single_roi=regions.loc[regions['Area']== single_region] # extracting individual region (e.g. Isocortex)
        zmaps[zmap_type][single_region]={} # preparing nested dictionary for an individual region
        for ind in single_roi['Cluster number']: # updating the dictionary for individual regions (e.g. ILA)
            zmaps_names = os.listdir(zmaps_path)
            zmaps[zmap_type][single_region][single_roi['Abbreviation'][ind-1]]={} 
            # one is substracted, since indexing starts from zero
            for cluster,cluster_name in enumerate(zmaps_names):
                zmaps[zmap_type][single_region][single_roi['Abbreviation'][ind-1]][f'cluster_{cluster+1}']={}
                for animals in os.listdir(os.path.join(zmaps_path,cluster_name[0])):           
                # upload zmaps for a selected cluster and the corresponding mask
                    zmap_cluster=nibabel.load(os.path.join(zmaps_path,cluster_name[0],\
                                                           animals)).get_fdata()
                        
                    #mask_cluster=nibabel.load(os.path.join(zmaps_clusters_path,cluster_name[1])).get_fdata()
                # take only a part of zmap that does not contain cluster, since these data will show 
                # uneterpritable zscores. this step is ommited for the correlational compuatation
                    # if cluster_substraction:
                    #     zmap_masked=substract_cluster_from_zmap(mask_cluster,zmap_cluster) 
                    #     # Substraction of the averaged mask
                    # else:    
                        
                    zmap_masked=zmap_cluster # for the correlation computation
                # no substraction for the similarity comparison
                #construc masks for individual brain regions according to the indixing from excel sheet
                #indices are from 1 till 165, 0 is a background. Since OLF, Hin/Midbrains are not present
                #not all 165 areas are present
                
                    mask_local=cluster_mask(ind,mask_labels_165_1_165) 
                #taking zscores corresponding to a brain region specified by local mask
                
                    zscore=zmap_masked[mask_local]
                #zscore[zscore==0]=float('NaN') # replace all zero elements with NaNs 
                #zscore=zscore[~np.isnan(zscore)] # remove all NaNs for further mean calculation
                #masks are chosen to match a specific ROI, functional data are uploaded to match clusters'
                
                    zmaps[zmap_type][single_region][single_roi['Abbreviation'][ind-1]][f'cluster_{cluster+1}']\
                   [f'animal_{animals[:animals.index("_")]}']={'zscores': zscore}    

with open('zmaps_sorted_per_condition.pkl', 'wb') as f:
    pickle.dump(zmaps, f)
    

# Sorted_per_cluster
zmaps={}
for zmap_type in zmap_type_condition: # go through the conditions stim or control
    zmaps_path=os.path.join(directory,f'Analysis_{zmap_type}','zstats') # data from individual animals
    zmaps[zmap_type]={} # creating a dictionary for further storage
    for cluster in os.listdir(os.path.join(zmaps_path)): # select the analyzed cluster
          zmaps[zmap_type][f'cluster_{cluster}']={}
          for animal in os.listdir(os.path.join(zmaps_path,cluster)): #select analyzed animal
            zmaps[zmap_type][f'cluster_{cluster}'][f'animal_{animal[:animal.index("_")]}']={}
            for single_region in regions_unique: # select ROI
                single_roi=regions.loc[regions['Area']== single_region] # extracting individual region (e.g. Isocortex)
                zmaps[zmap_type][f'cluster_{cluster}'][f'animal_{animal[:animal.index("_")]}'][single_region]={}
                for ind in single_roi['Cluster number']: #select subregion in roi (e.g BLA in "Amygdala")
                # updating the dictionary for individual regions (e.g. ILA)
                # cluster number corresponds to the numbering from 1 till 165
                #roi[single_region][single_roi['Abbreviation'][ind]]={'cluster_number':single_roi['Cluster number'][ind]}
                    zmaps[zmap_type][f'cluster_{cluster}'][f'animal_{animal[:animal.index("_")]}'][single_region]\
                        [single_roi['Abbreviation'][ind-1]]={}
                    zmap=nibabel.load(os.path.join(zmaps_path,cluster,\
                                                            animal)).get_fdata()    
                    mask_local=cluster_mask(ind,mask_labels_165_1_165)#upload mask for a specified region
                    zscore=zmap[mask_local]#apply the region mask for an individual zmap
                    zmaps[zmap_type][f'cluster_{cluster}'][f'animal_{animal[:animal.index("_")]}'][single_region]\
                        [single_roi['Abbreviation'][ind-1]]={'zscores': zscore}   

# Save the dictionary as pickle for the further use
with open('zmaps_sorted_per_cluster.pkl', 'wb') as f:
    pickle.dump(zmaps, f)