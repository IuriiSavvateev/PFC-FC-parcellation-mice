# -*- coding: utf-8 -*-
"""

@author: Iurii Savvateev
"""

## Adjusting labels

import os
import nibabel 
from pathlib import Path
import fnmatch

import shutil


# Computational modules
import numpy as np
# from sklearn.decomposition import KernelPCA
# from sklearn.decomposition import PCA
from sklearn import mixture # module containing Gaussian Mixture models
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering




#Plotting modules 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

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


def plot_dendrogram(model, **kwargs):
    """From sklearn tutorial. Create linkage matrix and then plot the dendrogram
     create the counts of samples under each node"""
     
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



def plot_two_charts(t,data1,data2,\
                    xlabel,data1_label,data2_label):
    """ Function plots two datasets on the same chart
    Inputs: t- axis, data1- first yaxis, data2- second yaxis
    xlabel- common xlabel axis, data1_label- label for the first yaxis, data2_label - label for the second yaxis
    Returns: plot
    """
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel(xlabel)
    ax1.set_xticks(t)
    ax1.set_ylabel(data1_label, color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(data2_label, color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def delete_all_folders(path):
    # Loop through all entries in the given directory
    for entry in listdir(path):
        full_path = join(path, entry)
        # Check if the entry is a directory
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f"Deleted existing folder to allow new data saving: {full_path}")


def mask_overlap(mask_1,mask_2):
    """Function that overlaps two masks
        Inputs: mask_1 (numpy array)
                mask_2 (numpy array)
        Returns: mask (boolean array)
    """
    x=np.zeros(mask_1.shape, dtype=int)
    z=np.argwhere(mask_1&mask_2==True)
    for i in range(len(z)):
        x[z[i][0]][z[i][1]][z[i][2]]=1
    return (x.astype(bool))


# Plotting individual bar charts
def plot_bar_regions(cluster_solution, region_analysed,data,cluster_list): #internal function of the module

#used in plot_all
    if np.all(data==0):
        return
    else:
        norm = [round(float(i)/sum(data)*100,1) for i in data] #transfer data to percentages
    
        def add_value_label(x_list,y_list): # specifying percentage above the bar chart
            for i in range(1, len(x_list)+1):
                plt.text(i-1,y_list[i-1]+1,y_list[i-1], ha="center")
    
        add_value_label(cluster_list,norm)   
        plt.bar(cluster_list, norm, width = 0.4)
        title=f'{cluster_solution+2} cluster solution. Composition of region {region_analysed}'
        plt.title(title)
        plt.xlabel("Raw Cluster number (may not match the manuscript)")
        plt.ylabel("Percentage")
        for pos in ['top', 'right']:
            plt.gca().spines[pos].set_visible(False)
        plt.savefig(f'{cluster_solution+2}-solution {region_analysed}-region.png')
        plt.close()  
        return

#Plotting bar charts for all solutions
def plot_all_regions(regions,statistics,n_solutions,parent,cluster_folder,cluster_list):
    for i in range(0,n_solutions): # loop through the cluster solutions (e.g 3 cluster solution)
        path = os.path.join(parent, cluster_folder.format(i+2))
        os.mkdir(path)
        os.chdir(path)
        for j in range(0,len(regions)):# loop through individual regions
        # is not counted, thus 1 till number of solutions. +2 comes from the shift and that 
        # two cluster solution is one variable, but contains two different clusters
            data=statistics[i][j,1:] #starting from 1 to not counting background
            plot_bar_regions(i,regions[j],data,cluster_list)
           
        os.chdir(parent)    
    

def anatomy_overlap_clusters(maskpath,mask_brain,mask_pfc,labels,savepath=[]):  
    """ Function describes each brain
    region in terms of present clusters
    Inputs: maskpath (str), mask_brain (numpy), mask_pfc (numpy), labels (numpy)
    Returns: plots in the parent folder foe each cluster solution, stat_total (numpy)
    and plots the resutls"""
    #mask_brain=mask
    
    os.chdir(maskpath)
    mask_new=nibabel.load('mask_Oh_V3_r_QBI_prefrontal_noinsula.nii.gz').get_fdata()
    a=np.unique(mask_new)[1:] #Exclude the zero cluster which is the background
    a=np.delete(a,np.argwhere(a==8)) #Remove FRP region, since it is not highly represented in the final
    
    count_cluster=[]
    cluster_labels = np.zeros(mask_brain.shape, dtype=int)
    roi=mask_overlap(mask_brain, mask_pfc) # region of interest, boolean
    compensation=roi.astype(int) # add 1 to all cluster numbers that are in the ROI, to have 
    #counting started at 1
#stat_total=np.zeros([1,len(a),6]) # For hierarchical clustering
    stat_total=np.zeros([len(labels),len(a),len(labels)+2]) # Third dimension corresponds to 12 clusters+ zero cluster
#stat_total=np.zeros(((len(kmeans),)*len(a),)*len(kmeans))
    for n in range(len(labels)): #loop through all labels
        cluster_labels[roi] = labels[n] #take the first cluster solution
        cluster_labels=cluster_labels+compensation #add 1
        x1=cluster_labels
        count_total=[]
        for i in range(len(a)): #loop through the regions, a contains list of the regions
            a1=np.argwhere(mask_new==a[i]) # indices of the desired region 
            count=[]
            for j in range(len(a1)): # loop through the indices in the mask that correspond to the region
                c=x1[a1[j][0]][a1[j][1]][a1[j][2]] # which cluster is in this region
                stat_total[n][i][c]+=1
                count.append(c)   
        count_total.append(count) 
    count_cluster.append(count_total)

    if not savepath:
        savepath=os.getcwd() #if the path is not specified save in the current directory
    else:
        pass
    
    regions = ['ACAd', 'ILA', 'ACAv', 'ORBL', 'ORBm',  'ORBvl', 'PL']
    cluster_folder = "{}_solution"
    n_solutions=stat_total.shape[0]
    statistics=stat_total
    cluster_list=[str(k+1) for k in range(n_solutions+1)] #create the list of cluster numbers
    plot_all_regions(regions,statistics,n_solutions,savepath,cluster_folder,cluster_list)
    
    return statistics

def nifti_save(niftipath,clustering_method,labels,mask_final,underlay_img,*,cluster_start=2):
    """Function that saves the results in nifti format
        Default: from which cluster number counting starts. Default=2
        Inputs: niftipath (str)
                clustering_method (str)
                labels (numpy array)
                mask_final (numpy array)
        Returns: None,Files are saved directly
    """
    cluster_labels = np.zeros(mask_final.shape, dtype=int)
    compensate=mask_final.astype(int) #increasing all cluster values in 1 (background stays 0)
    
    # since cluster numbering starts at zero
    path=Path(niftipath)
    path.mkdir(parents=True, exist_ok=True) 
    os.chdir(niftipath)
    
    # Load the Underlay_QBI.nii to retrieve the geometry (affine and header)
    affine = underlay_img.affine
    header = underlay_img.header
    

    for idx,n in enumerate(labels):
        cluster_labels[mask_final]=n
        cluster_labels=cluster_labels+compensate
        image = nibabel.Nifti1Image(cluster_labels, affine, header=header)
        nibabel.save(image,f'{clustering_method}-{idx+2}.nii.gz') #first element correspond to two cluster solution
        
        
# def labels_adjust(labels_old,adjust_dictionary):
#     """ Function substitutes label numbers obtained during clustering to the numbers matching the 
#     order in labels in 60% data solution. The form of the cluster does not change, only the exact 
#     numbers assigned to each cluster. It allows to use these labels for a consistent characterization
#     of the anatomical regions with respect to the present clusters, as well as the analysis of the cluster
#     anatomy. Dictionary is manually created by comparing the cluster numbers in fsleyes 
#     Inputs: labels_old (list)- old labels, adjust_dictionary(dictionary) with old label:new label structure
#     In the dictionary put the new cluster numbers as strings to avoid changing indices back, when the swap
#     is performed
#     Return: Labels_adjusted (np.array) - new labels in accordance to the order chosen for 60% data solution
#     """ 
#     labels_new=[adjust_dictionary.get(item,item) for item in labels_old]
#     labels_new=list(map(int,labels_new)) # transform all elements to integers
#     #labels_new=
#     return np.array(labels_new)

############################################# Beginning of main file
def main(data_type='pfc-pfc',data_size=60, classification_method='kmeans'):
    #default parameters as in the main figure
    
    
    # Choose the corresponding dictionary for the labels adjustment, if the other
    # conditon (not pfc-pfc, 60) is used. 
    # Labels are adjusted to match the chosen numbering in pfc-pfc 60
    
    
    ## dictionary={5:'7',7:'5'} # for kmeans pfc-pfc 20% nifti vs new (as in pfc-pfc 60)
    # # dictionary={1:'5',2:'6',3:'2',4:'4',5:'1',6:'3',7:'7'} # for kmeans pfc-pfc 40% 
    # # dictionary={1:'1',2:'2',3:'7',4:'3',5:'4',6:'5',7:'6'} # for kmeans pfc-pfc 80% 
    
    # # dictionary={1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7'} # for kmeans pfc-pfc 100% 
    # # dictionary={1:'3',2:'5',3:'7',4:'4',5:'6',6:'1',7:'2'} # for gmm pfc-pfc 100% 
    # # dictionary={1:'1',2:'7',3:'4',4:'6',5:'3',6:'5',7:'2'} # for Hierarchical pfc-pfc 100%
    
    # # dictionary={1:'1',2:'2',3:'4',4:'3'} # for kmeans pfc-all 100%
    # # dictionary={1:'2',2:'4',3:'1',4:'3'} # for gmm pfc-all 100%
    # # dictionary={1:'2',2:'1',3:'3',4:'4'} # for hierarchical pfc-all 100%


    cluster_solution=7
    directory='D:\Parcellation\Github\Clustering'
    area='brain' # choose  global area (e.g. Global mask)

    os.chdir(directory)    
    # importing filenames, global mask, data and maskpath
    filenames,mask,datapath,maskpath=load(directory,area)  

# Converting mask to boolean and save the list of analyzed files
    mask = mask.astype(bool) #global mask

# Selecting a mask for the target area
    os.chdir(maskpath)
    mask_pfc=nibabel.load('mask_prefrontal_cortex_noinsula.nii.gz').get_fdata()
    mask_pfc = mask_pfc.astype(bool)

#Overkapping a global and local masks
    mask_brain_pfc=mask_overlap(mask, mask_pfc)

    
#Uploading the precomputed correlation matrix 

    parent='D:\Parcellation\Github\Clustering'
    corrpath=os.path.join(parent, "Mean_correlation_matrix",f'{data_type}')
    os.chdir(corrpath)
    corr_name=fnmatch.filter(os.listdir('.'), f'mean_correlation_*_{data_size}.npy')
    mean_correlation_matrix=np.load(corr_name[0]) # 
    os.chdir(parent)

    
#Performing clustering of the mean_correlation_matrix
    
    n_clusters=cluster_solution+1
    X_embedded=mean_correlation_matrix # potentially one can do dimensionality 
    #reduction before clustering. Explored by a separte script, see TSNE-PCA

    if classification_method == 'gmm': # Not for GMM the computations may take long

        
        results=[mixture.GaussianMixture(n, covariance_type='full', random_state=0, init_params='kmeans')\
           .fit(X_embedded) for n in range(2,n_clusters)] # fix seed at 0
    
        
        labels=[n.predict(X_embedded) for n in results]
        
            
        # If needed AIC and BIC are extracted. Not used.
        bic_gmm=[m.bic(X_embedded) for m in results]
        aic_gmm=[m.aic(X_embedded) for m in results]
    

    elif classification_method == 'kmeans':
       
        results=[KMeans(n,random_state=50).fit(X_embedded) for n in range(2,n_clusters)] # kMeans clustering
        
        labels=[n.labels_ for n in results]
        score=[silhouette_score(X_embedded,i.labels_,
                              metric='euclidean') for i in results]
        inertia=[i.inertia_ for i in results ]
        
        plot_two_charts(np.arange(3,n_clusters), score[1:], inertia[1:],\
                        'Number of clusters','Average Silhouette score', "Inertia")
            
    elif classification_method == 'Hierarchical': 
      
        results=[AgglomerativeClustering(n_clusters=n, linkage="ward",compute_distances=True).fit(X_embedded) \
                 for n in range(2,n_clusters)] #ward linkage is used
            
        labels=[n.labels_ for n in results]
        # plot the top three levels of the dendrogram for seven cluster solution
        plot_dendrogram(results[5], truncate_mode="lastp", p=7) # minus two since staring from the second cluster
        plt.xlabel("Cluster number (voxels)")
        plt.ylabel("Linkage")
        plt.show()
      
  
 
     
     #Saving corresponding niftifiles (already done for kmeans)
    underlay = nibabel.load(join(parent,'Masks',"Underlay_QBI.nii"))
    titlename=f'{data_type}_{data_size}_{classification_method}'
    
    niftipath=join(parent,'Clusters_Nifti',titlename)
    path=Path(niftipath)
    path.mkdir(parents=True, exist_ok=True) 
    nifti_save(niftipath, titlename, labels, mask_brain_pfc, underlay)
    

    # labels_corr=[i+1 for i in labels] # labels_corr and labels_fit_corr are used to adjust indexings
    # #adjusting back is necessary since the adjustment is present inside anatomy_overlap_clusters
    # labels_fit=[labels_adjust(i,dictionary) if ind==(cluster_solution-2) else i for ind,i in enumerate(labels_corr)]
    # labels_fit_corr=[i-1 for i in labels_fit]
    
    plotpath='D:\Parcellation\Github\Clustering\Anatomy'
    delete_all_folders(plotpath)
    #plotpath=os.path.join(parent,'Plots',f'{classification_method}',f'{data_type}',f'{data_size}')
    path=Path(plotpath)
    path.mkdir(parents=True, exist_ok=True) 
    os.chdir(plotpath)

    anatomy_overlap_clusters(maskpath,mask,mask_pfc,labels,plotpath) 

if __name__=='__main__':
    main()
