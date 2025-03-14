Folder contains scripts to perform the clustering form the rsfMRI images. 
Refer to 10.5281/zenodo.14989998 for the complete dataset. 

1.Put the dataset in "Data" folder, currently it contains no data due to the size restrictions. 001_FIXed_QBI.nii.gz example file is stored in Git Large File Storage

2.Open correlation_individual.py and adapt the path stored in "directory" variable. By default it computes pfc-pfc correlation matrix for the first animal. If you wish to change these settings adapt the corresponding variables 
in lines 100 and 101, or use external outputs options in lines 104-105. 

3.correlation_individual.py saves the correlation matrix for the chosen animal in Correlation-matrix folder in .npy format (an example matrix for the first animal, for pfc-pfc condition is stored)

3.1. If you wish to compute the correlation matrices for all data stored in "Data" folder and all conditions (e.g. pfc-pfc, pfc-nopfc, pfc-brain), use the correlation_individual_run.sh (do not forget to adapt correlation_individual.py accordingly, see (2)) 

4. At the next stage, we (a) subselect part of the data (e.g. 20, 40, 60, 80 or 100), (b) compute the mean correlation matrix for the selected data, (c) perform kmeans clustering and save corresponding silhouette and inertia 
scores and then repeat steps (a)-(c) 200 times, thereby, collecting silhouette and inertia data displayed in the manuscript at Fig 1, 2 and S1, S3, S4. Due to the computational demand of the procedure the original codes executed in the ETHZ Euler cluster are provided: Euler_correlation_combined.sh and Euler_correlation_combined.py. Results are stored in "Iteration" folder.

*Note that when scripts are run from the beginning the exact assignments of the numbers to clusters may differ, therefore, to reproduce the exact match with the manuscript (e.g. C1 - C4 in Fig 1,2 and supplementary) one may need to adjust the names of the resulted clusters based on the cluster images (for instance, by checking fsleyes).

5. To compute an overlap between the clusters and the Allen-based anatomical parcellation an example of a mean correlaton matrix computed at step 4 was used. An example of the mean correlation matrix for pfc-pfc condition with 60 animals is uploaded to "Mean_correlation_matrix" folder. This data was used to compute clustering displayed in Fig 2F. Results of an anatomy overlap are saved in "Anatomy folder", clusters in nii.gz format are saved in "Clusters_Nifti" folder. 
In a separate folder named "all_conditions_manuscript" we provide nifti files for all clusters analysed in the manuscript.

6. We summarize the anatomy analysis (from step 5) in one chart (Fig 1F and Fig 2F) using clusters_anatomy_plots.py. The data is inserted directly in the script, one should uncomment the relevant condition. We provide data for all conditions shown in the paper. 

7. Finally, We probe with TSNE_PCA two different dimensionality reduction techniques: TSNE and PCA to get a better understanding of the data structure. Results are saved in TSNE_PCA folder The examples of those representations are shown at Fig 1D and Fig 2D



