"""
@author: Iurii Savvateev
"""


from scipy.spatial.distance import cityblock
import seaborn as sns

import parcellation_module_ts_ss as ts_ss_comparison

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import os
from scipy import stats



data_path=r'D:\Parcellation\Github\Cluster_perturbation_analysis\Cluster_statistics'
plots_path=r'D:\Parcellation\Github\Cluster_perturbation_analysis\Matrices'

os.chdir(data_path)
statistical_method='Pearson Correlation' #Inverse TS_SS, Pearson Correlation or Manhattan

compared_conditions=['GFP','Gi']


# Using a separate module to compute TS-SS scores
def statistics_TS_SS(x,y):
    vec1=np.nanmean(x,axis=0)
    vec2=np.nanmean(y,axis=0)
    return ts_ss_comparison.TS_SS(vec1[~np.isnan(vec1)],vec2[~np.isnan(vec2)])
 

pvalue_matrix=np.zeros((4,4))
pvalue_matrix_corr=np.zeros((4,4))
score_matrix=np.zeros((4,4))
mask_off_diag=~np.diag(np.ones(4)).astype(bool)

# Those areas are excluded, since the signal was not detected there or was not 
# a part of the analysis 

exclude_abbreviations = ['PAR', 'ENTl', 'ENTm', 'POST', 'PRE', 'IA', 'PMd', 'SUM']


cluster_plot_order=[1,2,3,4] #ACA,PLc, ILc, ORBc
for ind_0,cluster_0 in enumerate(cluster_plot_order):
    for ind_1,cluster_1 in enumerate(cluster_plot_order):

        data_0=pd.read_excel(os.path.join(data_path,\
                                  f'{compared_conditions[0]}_masked_cluster_{cluster_0}.xlsx'))
        data_1=pd.read_excel(os.path.join(data_path,\
                                  f'{compared_conditions[1]}_masked_cluster_{cluster_1}.xlsx')) 

        
        # For some reason Gozzi template excludes the regions listed in exclude_abbreviations
        data_0=data_0[~data_0['Abbreviation'].isin(exclude_abbreviations)]
        data_1=data_1[~data_1['Abbreviation'].isin(exclude_abbreviations)]
                                                   
        data_0_values=data_0[data_0.columns[data_0.columns.str.contains(pat='animal')]].T.to_numpy()
        data_1_values=data_1[data_1.columns[data_1.columns.str.contains(pat='animal')]].T.to_numpy()
        
        
        vec0=np.nanmean(data_0_values,axis=0)
        vec1=np.nanmean(data_1_values,axis=0)
        
        if statistical_method == 'Manhattan':
            score_matrix[ind_0,ind_1]=cityblock(vec0[~np.isnan(vec0)],\
                                              vec1[~np.isnan(vec1)])
        elif statistical_method == "Pearson Correlation":
            score_matrix[ind_0,ind_1]=stats.pearsonr(vec0[~np.isnan(vec0)],\
                                            vec1[~np.isnan(vec1)])[0]
            pvalue_matrix_corr[ind_0,ind_1]=stats.pearsonr(vec0[~np.isnan(vec0)],\
                                                  vec1[~np.isnan(vec1)])[1]
        elif statistical_method == "Inverse TS_SS":

            score_matrix[ind_0,ind_1]=1/statistics_TS_SS(data_0_values,data_1_values)


# normalization
score_matrix=np.divide(score_matrix,score_matrix.max())
cluster_names=['C1','C2','C3','C4']


# Plotting part 

df=pd.DataFrame(score_matrix,columns=cluster_names)
color_palette=sns.color_palette("light:b", as_cmap=True)

ax=plt.axes()
sns.heatmap(df,cmap='Blues', #Oranges
            linewidths=.5,yticklabels=cluster_names,ax=ax)    


ax.set_yticklabels(cluster_names,fontsize=13)
ax.set_xticklabels(cluster_names,fontsize=13)

ax.set_ylabel(compared_conditions[0],fontsize=15, weight='bold')
ax.set_xlabel(compared_conditions[1],fontsize=15, weight='bold')


# within if-else loop put the labels for the charts


if statistical_method =="Inverse TS_SS":
    ax.set_title(f'Cluster similarity (Inverse TS_SS score):\n {compared_conditions[0]} vs  {compared_conditions[1]}',fontsize=15, weight='bold')
                  
if statistical_method =="Manhattan":
    ax.set_title(f'Cluster similarity (Manhattan distance):\n {compared_conditions[0]} vs  {compared_conditions[1]}',fontsize=15, weight='bold')    
                  
if statistical_method =="Pearson Correlation":
    ax.set_title(f'Cluster similarity (Pearson Correlation):\n {compared_conditions[0]} vs  {compared_conditions[1]}',fontsize=15, weight='bold')


similarity_matrix=ax.get_figure()
similarity_matrix.savefig(os.path.join(plots_path,f'{compared_conditions[0]}_{compared_conditions[1]}_{statistical_method}.png'))



        
