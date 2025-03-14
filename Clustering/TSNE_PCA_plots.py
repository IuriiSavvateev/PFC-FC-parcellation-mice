# -*- coding: utf-8 -*-
"""

@author: Iurii Savvateev
"""
## Adjusting labels

import os
import fnmatch

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap

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


##################### 4 cluster solution ####################################

data_type='pfc-all'
data_size=60
classification_method='kmeans'
n_clusters=8 # the max will be 7 cluster solution

parent='D:\Parcellation\Github\Clustering'
figpath=os.path.join(parent,'TSNE_PCA')
corrpath=os.path.join(parent, "Mean_correlation_matrix",f'{data_type}')
corr_name=fnmatch.filter(os.listdir(corrpath), f'mean_correlation_*_{data_size}.npy')
mean_correlation_matrix=np.load(os.path.join(corrpath,corr_name[0])) # 

# Perform K-means clustering 
X_embedded=mean_correlation_matrix # no embedding  for clustering

results=[KMeans(n,random_state=50).fit(X_embedded) for n in range(2,n_clusters)] # kMeans clustering

labels=[n.labels_ for n in results]
score=[silhouette_score(X_embedded,i.labels_,
                      metric='euclidean') for i in results]
inertia=[i.inertia_ for i in results ]
#np.save('score_kmeans',np.array(score))
##np.save('inertia_kmeans',np.array(inertia))

plot_two_charts(np.arange(3,n_clusters), score[1:], inertia[1:],\
                'Number of clusters','Average Silhouette score', "Inertia")


# define colour schema for 4 cluster solution 

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
red = np.array([1, 0, 0, 1])
violet = np.array([0.8, 0, 0.9, 1])
blue=np.array([0, 191/255, 1, 1])
orange=np.array([1, 0.6, 0.1, 1])

part=int(0.25*len(newcolors))


newcolors[0:part, :] = orange
newcolors[part:2*part,:] = blue
newcolors[2*part:3*part, :] = violet
newcolors[3*part:4*part, :] = red

newcmp_4 = ListedColormap(newcolors)

 
data_plot=labels[2] # take 4 cluster solution

######### TSNE #########
X_TSNE = TSNE(n_components=3, init='pca').fit_transform(X_embedded) 

plt.scatter(X_TSNE[:,2], X_TSNE[:,0], c=data_plot, cmap=newcmp_4) 

plt.xlabel("Third component",fontsize=20,fontweight="bold")
plt.ylabel("First component",fontsize=20,fontweight="bold")
plt.xticks(rotation='horizontal', fontsize=15)
plt.yticks(rotation='horizontal', fontsize=15)
plt.title(f'2D projection of 3D t-SNE(k={len(np.unique(data_plot))})',fontsize=20,fontweight="bold")
plt.savefig(os.path.join(figpath,f'{data_type}_{data_size}_{classification_method}_cluster_{len(np.unique(data_plot))}_2D_TSNE.png'), transparent=True)
  
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')


ax.scatter(X_TSNE[:,0], X_TSNE[:,1], X_TSNE[:,2], c=data_plot, cmap=newcmp_4)
ax.set_xlabel("First component",fontsize=25,fontweight="bold",labelpad = 20)
ax.set_ylabel("Second component",fontsize=25,fontweight="bold",labelpad = 20)
ax.set_zlabel("Third component",fontsize=25,fontweight="bold",labelpad = 20)
ax.tick_params(labelsize=15)
ax.set_title(f'3D t-SNE(k={len(np.unique(data_plot))})',fontsize=30,fontweight="bold")
plt.savefig(os.path.join(figpath,f'{data_type}_{data_size}_{classification_method}_cluster_{len(np.unique(data_plot))}_3D_TSNE.png'), transparent=True)

######### PCA #########
pca=PCA(n_components=4).fit(mean_correlation_matrix) # PCA
X_embedded=pca.transform(mean_correlation_matrix)

plt.figure()
plt.scatter(X_embedded.T[0,:],X_embedded.T[1,:], c=data_plot, cmap=newcmp_4)
plt.xlabel('First component',fontsize=20,fontweight="bold")
plt.ylabel("Second component",fontsize=20,fontweight="bold")
plt.xticks(rotation='horizontal', fontsize=15)
plt.yticks(rotation='horizontal', fontsize=15)
plt.title(f'PCA (k={len(np.unique(data_plot))})',fontsize=20,fontweight="bold")
plt.savefig(os.path.join(figpath,f'{data_type}_{data_size}_{classification_method}_cluster_{len(np.unique(data_plot))}_PCA.png'), transparent=True)

##################### 7 cluster solution ####################################

data_type='pfc-pfc'
data_size=60
classification_method='kmeans'
n_clusters=8 # the max will be 7 cluster solution

parent='D:\Parcellation\Github\Clustering'
figpath=os.path.join(parent,'TSNE_PCA')
corrpath=os.path.join(parent, "Mean_correlation_matrix",f'{data_type}')
corr_name=fnmatch.filter(os.listdir(corrpath), f'mean_correlation_*_{data_size}.npy')
mean_correlation_matrix=np.load(os.path.join(corrpath,corr_name[0])) # 

# Perform K-means clustering 
X_embedded=mean_correlation_matrix # no embedding  for clustering

results=[KMeans(n,random_state=50).fit(X_embedded) for n in range(2,n_clusters)] # kMeans clustering

labels=[n.labels_ for n in results]
score=[silhouette_score(X_embedded,i.labels_,
                      metric='euclidean') for i in results]
inertia=[i.inertia_ for i in results ]
#np.save('score_kmeans',np.array(score))
##np.save('inertia_kmeans',np.array(inertia))

plot_two_charts(np.arange(3,n_clusters), score[1:], inertia[1:],\
                'Number of clusters','Average Silhouette score', "Inertia")



# define colour schema for 7 cluster solution

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
red = np.array([1, 0, 0, 1])
violet = np.array([0.8, 0, 0.9, 1])
blue=np.array([0, 191/255, 1, 1])
dark_blue=np.array([63/255, 72/255, 204/255, 1])
pink=np.array([239/255, 136/255, 190/255, 1])
yellow=np.array([212/255, 175/255, 55/255, 1])
orange=np.array([1, 0.6, 0.1, 1])

part=int((1/7)*len(newcolors))


newcolors[0:part, :] = red
newcolors[part:2*part,:] = blue #(blue)
newcolors[2*part:3*part, :] = dark_blue #(dark_blue)
newcolors[3*part:4*part, :] = orange #(orange)
newcolors[4*part:5*part, :] = yellow #(yellow)
newcolors[5*part:6*part, :] = violet #(pink)
newcolors[6*part:, :] = pink #(violet)

newcmp_7 = ListedColormap(newcolors)

data_plot=labels[5] # take 7 cluster solution (labels[n-2] for n cluster solution)


######### TSNE #########
X_TSNE = TSNE(n_components=3, init='pca').fit_transform(X_embedded) 

plt.scatter(X_TSNE[:,2], X_TSNE[:,0], c=data_plot, cmap=newcmp_7) 

plt.xlabel("Third component",fontsize=20,fontweight="bold")
plt.ylabel("First component",fontsize=20,fontweight="bold")
plt.xticks(rotation='horizontal', fontsize=15)
plt.yticks(rotation='horizontal', fontsize=15)
plt.title(f'2D projection of 3D t-SNE(k={len(np.unique(data_plot))})',fontsize=20,fontweight="bold")
plt.savefig(os.path.join(figpath,f'{data_type}_{data_size}_{classification_method}_cluster_{len(np.unique(data_plot))}_2D_TSNE.png'), transparent=True)
  
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')


ax.scatter(X_TSNE[:,0], X_TSNE[:,1], X_TSNE[:,2], c=data_plot, cmap=newcmp_7)
ax.set_xlabel("First component",fontsize=25,fontweight="bold",labelpad = 20)
ax.set_ylabel("Second component",fontsize=25,fontweight="bold",labelpad = 20)
ax.set_zlabel("Third component",fontsize=25,fontweight="bold",labelpad = 20)
ax.tick_params(labelsize=15)
ax.set_title(f'3D t-SNE(k={len(np.unique(data_plot))})',fontsize=30,fontweight="bold")
plt.savefig(os.path.join(figpath,f'{data_type}_{data_size}_{classification_method}_cluster_{len(np.unique(data_plot))}_3D_TSNE.png'), transparent=True)

######### PCA #########
pca=PCA(n_components=4).fit(mean_correlation_matrix) # PCA
X_embedded=pca.transform(mean_correlation_matrix)

plt.figure()
plt.scatter(X_embedded.T[0,:],X_embedded.T[1,:], c=data_plot, cmap=newcmp_7)
plt.xlabel('First component',fontsize=20,fontweight="bold")
plt.ylabel("Second component",fontsize=20,fontweight="bold")
plt.xticks(rotation='horizontal', fontsize=15)
plt.yticks(rotation='horizontal', fontsize=15)
plt.title(f'PCA (k={len(np.unique(data_plot))})',fontsize=20,fontweight="bold")
plt.savefig(os.path.join(figpath,f'{data_type}_{data_size}_{classification_method}_cluster_{len(np.unique(data_plot))}_PCA.png'), transparent=True)





