# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 16:15:37 2022

@author: Iurii Savvateev
"""
#from numba import jit
from sklearn.preprocessing import Normalizer
import os
#import glob2
import nibabel # module to read nifti files
import matplotlib.pyplot as plt 
import numpy as np

from os import listdir
from os.path import isfile, join


#Loading of the data
def load (directory,area):
    working_directory=directory
    os.chdir(working_directory)

#Specifying data and masks directory
    datapath=working_directory+'\Data_50'  # Data_50 or Data or Data_100
    maskpath=working_directory+'\Masks'

#Getting data
    os.chdir(datapath)
    #filenames_nii = glob2.glob('*.nii' )
    #filenames_nii_gz = glob2.glob('*.gz' )
    #filenames=filenames_nii+filenames_nii_gz
    filenames=[f for f in listdir(datapath) if isfile(join(datapath, f))]
    filenames = sorted(filenames)

#Getting mask for global area
    os.chdir(maskpath)
    if area == 'hypothalamus':
        mask=nibabel.load('QBI_annotation_hypothalamus.nii').get_fdata()
    elif area == 'hippocampus':
        mask=nibabel.load('QBI_annotation_hippocampus.nii').get_fdata()
    elif area =='striatum':
        mask=nibabel.load('QBI_annotation_striatum.nii.gz').get_fdata()
    elif area == 'isocortex':
        mask=nibabel.load('QBI_annotation_isocortex.nii.gz').get_fdata()
    elif area == 'pfc_noinsula':
        mask=nibabel.load('mask_prefrontal_cortex_noinsula.nii.gz').get_fdata()
    elif area == 'brain':
        mask=nibabel.load('QBI_brain.nii.gz').get_fdata()
    elif area == 'brain_noiso':
        mask=nibabel.load('QBI_brainwithoutisocortex.nii.gz').get_fdata()
    else:
        print ("No such masks "+area+" is available")
        mask=[]
    os.chdir(working_directory)
    return filenames, mask, datapath, maskpath

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
#Computing functional connectivity

def connectivity(metric,form,data,mouse,index_target,index_nontarget):
    connectome_measure = ConnectivityMeasure(kind=metric)
    timeseries_each_subject = data[mouse]
    conn = connectome_measure.fit_transform([timeseries_each_subject])
    conn=np.concatenate(conn, axis=0)[index_target]
    if form =='target-target':
        conn=conn[:,index_target]
    elif form == 'target-nontarget':    
        conn=conn[:,index_nontarget]
    return conn

# Plotting individual bar chartsde
def plot_bar(cluster_solution, cluster_analysed,data,regions): #internal function of the module

#used in plot_all
    if np.all(data==0):
        return
    else:
        norm = [round(float(i)/sum(data)*100,1) for i in data] #transfer data to percentages
    
        def add_value_label(x_list,y_list): # specifying percentage above the bar chart
            for i in range(1, len(x_list)+1):
                plt.text(i-1,y_list[i-1]+1,y_list[i-1], ha="center")
    
        add_value_label(regions,norm)   
        plt.bar(regions, norm, width = 0.4)
        title="{} cluster solution. Anatomy of cluster {}".format(cluster_solution+2, cluster_analysed)
        plt.title(title)
        plt.xlabel("Region")
        plt.ylabel("Percentage")
        for pos in ['top', 'right']:
            plt.gca().spines[pos].set_visible(False)
        plt.savefig('{}-solution {}-cluster.png'.format(cluster_solution+2, cluster_analysed))
        plt.close()  
        return

#Plotting bar charts for all solutions
def plot_all(regions,statistics,n_solutions,parent,cluster_folder):
    for i in range(n_solutions): # loop through the cluster solutions (e.g 3 cluster solution)
        path = os.path.join(parent, cluster_folder.format(i+2))
        os.mkdir(path)
        os.chdir(path)
        for j in range(1,n_solutions+2):# loop through individual clusters, cluster zero
        # is not counted, thus 1 till number of solutions. +2 comes from the shift and that 
        # two cluster solution is one variable, but contains two different clusters
            data=statistics[i][:,j]
            plot_bar(i,j,data,regions)
        os.chdir(parent)
#Plotting bar charts for individual regions
