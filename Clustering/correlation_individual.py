# -*- coding: utf-8 -*-
"""
@author: Iurii Savvateev
"""

from sklearn.preprocessing import Normalizer
import os
import sys
import nibabel 
import numpy as np


from os import listdir
from os.path import isfile, join



#Loading of the data
def load (directory,area):
    working_directory=directory

#Specifying data and masks directory
    datapath=join(working_directory,'Data') # Data should be put here
    maskpath=join(working_directory,'Masks')

#Getting data

    filenames=[f for f in listdir(datapath) if isfile(join(datapath, f))]
    filenames = sorted(filenames)

#Getting mask for global area

    if area == 'pfc_noinsula':
        mask=nibabel.load(join(maskpath,'mask_prefrontal_cortex_noinsula.nii.gz')).get_fdata()
    elif area == 'brain':
        mask=nibabel.load(join(maskpath,'QBI_brain.nii.gz')).get_fdata()
    elif area == 'brain_noiso':
        mask=nibabel.load(join(maskpath,'QBI_brainwithoutisocortex.nii.gz')).get_fdata()
    else:
        print (f"No masks for {area} is available")
        mask=[]
    os.chdir(working_directory)
    return filenames, mask,datapath, maskpath

def corr_area(mask_target,mask_global):
    mask=mask_global
    mask_pfc=mask_target
    mask_iso_line=mask.reshape(len(mask)*len(mask[0])*len(mask[0][0])) #mask global in line
    mask_pfc_line=mask_pfc.reshape(len(mask_pfc)*len(mask_pfc[0])*len(mask_pfc[0][0])) # mask target in line

    index_iso=np.argwhere(mask_iso_line==True) #mask global True voxels
    index_pfc=np.argwhere(mask_pfc_line==True) #mask target True voxels

    index_iso_pfc=np.nonzero(np.in1d(index_iso, index_pfc))[0] # overlap global=>target
    index_iso_nopfc=np.where(np.in1d(index_iso, index_pfc)==0)[0] #no overlap global=>no target
    return index_iso_pfc, index_iso_nopfc

#Getting functional data
def fdata (filenames,mask,mouse):
    func_data = nibabel.load(filenames[mouse]).get_fdata()
    X = func_data[mask].T
    transformer = Normalizer().fit(X) #apply L2 norm
    X = transformer.transform(X)
    return(X)

from nilearn.connectome import ConnectivityMeasure
#Computing correlational functional connectivity

def connectivity(metric,data,index_target,index_nontarget,form=None): 
    # for an individual animal
    connectome_measure = ConnectivityMeasure(kind=metric)
    timeseries_each_subject = data
    conn = connectome_measure.fit_transform([timeseries_each_subject])
    conn=np.concatenate(conn, axis=0)[index_target]
    # taking only subpart of the data if want to compute pfc-pfc, pfc-nopfc
    # by default computing pfc-brain
    if form =='pfc-pfc':
        conn=conn[:,index_target]
    elif form == 'pfc-nopfc':    
        conn=conn[:,index_nontarget]
    return conn


directory='D:\Parcellation\Github\Clustering'
area='brain' # choose analyzed area


filenames,mask, datapath ,maskpath=load(directory,area)  

# Converting mask to boolean and save the list of analyzed files
mask = mask.astype(bool) #global mask (e,g, brain)
mouse_list=list(range(len(filenames)))

# Selecting a mask for the target area (e.g. prefrontal cortex)
mask_pfc=nibabel.load(join(maskpath,'mask_prefrontal_cortex_noinsula.nii.gz')).get_fdata()
mask_pfc = mask_pfc.astype(bool)


# example call 
#animal_number=0
#corr_type='pfc-nopfc' #pfc-pfc, pfc-nopfc, pfc-brain as default

# Specifying external parameters for a bash script call
corr_type=sys.argv[1] #pfc-pfc, pfc-nopfc, brain as default
animal_number=int(sys.argv[2])


# Construction an overlaping mask for the data extraction
index_iso_pfc, index_iso_nopfc=corr_area(mask_pfc,mask)
print('Functional data extraction is launched')

os.chdir(datapath)
dmap1=fdata(filenames,mask,animal_number)

print('Functional data extraction is done')

# Computing correlations and mean correlation matrix
print('Correlation computation is launched')
os.chdir(join(directory, 'Correlation-matrix'))

correlation=connectivity('correlation',dmap1,index_iso_pfc,index_iso_nopfc,form=corr_type) 
np.save(f'{animal_number+1}_{corr_type}',correlation)       

print('Correlation computation is done')