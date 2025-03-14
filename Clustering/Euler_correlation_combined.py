# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:30:02 2022
@author: Iurii Savvateev
"""

## Adjusting labels

import os
import sys
from pathlib import Path

#import itertools

# Computational modules
import numpy as np
import random
#from sklearn.decomposition import KernelPCA
#from sklearn.decomposition import PCA
#from sklearn import mixture # module containing Gaussian Mixture models
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from sklearn.cluster import AgglomerativeClustering

# Module for computing melodic overlap with the zscores
#import parcellation_module_ts_ss as ts_ss_comparison
corr_type=sys.argv[1] 
data_part=int(sys.argv[2])

#corr_type='all'
#data_part=20
method='kmeans'

parent='/cluster/work/wenderoth/iuriis/Parcellation/October-2022'
corrpath=os.path.join(parent, "corr_test_100",f"pfc-{corr_type}")
savepath=os.path.join(parent, 'corr_test_100','iteration', f"pfc-{corr_type}")

path=Path(savepath)
path.mkdir(parents=True, exist_ok=True)

os.chdir(corrpath)
names=os.listdir('.')
correlations=[np.load(i) for i in names]

n_clusters=11 #specify the max cluster number (excluded), thus for 5 cluster solution specify 6
# clusters 3,4,5 will be computed
max_iteration=200
combined_scores=np.empty([max_iteration+1,n_clusters-2]) # create matrix to collect silhouette scores
# n_clusters -2, since we start including clusters from 3
# max_iteration +1, since iteration starts from zero, therefor need, for instance, 4 rows for max_iteration 3
combined_inertia=np.empty([max_iteration+1,n_clusters-2])
iteration=0
while iteration <= max_iteration:
    
    print(f'iteration {iteration} from {max_iteration} is in the process')
    selected_data=np.array(random.sample(range(0,100),data_part))
    correlations_selected=[correlations[i] for i in selected_data]
    mean_correlation_matrix=np.mean(correlations_selected,axis=0)
    results=[KMeans(n,random_state=50).fit(mean_correlation_matrix) for n in range(2,n_clusters)]
    score=[silhouette_score(mean_correlation_matrix,i.labels_,metric='euclidean') for i in results]
    inertia=[i.inertia_ for i in results]
    combined_scores[iteration,:]=score
    combined_inertia[iteration,:]=inertia
    iteration +=1
    
# mean_combined_scores=np.mean(combined_scores, axis=0)
# std_combined_scores=np.std(combined_scores, axis=0)
# sem_combined_scores=std_combined_scores/np.sqrt(combined_scores.shape[1])

os.chdir(savepath)
np.savez(f'{method}_pfc_{corr_type}_{data_part}',inertia=np.array(combined_inertia),\
         silhouette_score=np.array(combined_scores))

