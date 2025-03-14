Folders "Analysis_GFP_masked" and "Analysis_Gi_masked" correspond to the fMRI recordings during DREADD active time window from the mice injected with AAV8-hSyn-GFP (n=19) and inhibitory DREADD AAV8-hSyn-hM4Di (n=15) viruses. 

Fore details see 
Rocchi, F. et al. Increased fMRI connectivity upon chemogenetic inhibition of the mouse prefrontal cortex. Nature Communications 13, 1056 (2022). https://doi.org/10.1038/s41467-022-28591-3 

Contact the corresponding author from Rocchi et al. for the raw data. We use the following pipeline to analyse the data from Rocchi et al. to ultimately get zmaps stored in "Analysis_GFP_masked" and "Analysis_Gi_masked". 

1.Create individual masks (from KMEANS-4-QBI_r_ref.nii.gz). For instance, fslmaths KMEANS-4-QBI_r_ref.nii.gz -thr 2.5 -uthr 3.5 -bin KMEANS-4-QBI_r_ref_3.nii.gz. Results are stored in 
...\Github\Histological_evaluation\visualization\Individual_cluster_masks. 

2. Overlap masks denoting analysed brain area with the individual data. fslmaths "$data_file" -mul "$mask_file" "$output_file". Used mask (QBI_brain_new_Rocchi_Template.nii.gz) is stored in ...\Github\Cluster_perturbation_analysis\Brain_mask

3. Computing average timeseries from regions defined by individual masks computed at step (1) and then use these timeseries to compute zmaps using fsl_glm. These two steps are summarized in zmaps_compute.sh file. Note that you need to adapt the paths to run the script.

4. The result of step (3) folder are stored in subfolders: "timeseries" and "zstats" (parts of  "Analysis_GFP_masked" and "Analysis_Gi_masked")

5. At the next step we use zmaps_calculation.py to sort the z-statistics from step (4) into two dictionaries: zmaps_sorted_per_condition.pkl and zmaps_sorted_per_cluster.pkl. These dictionaries save the statistics sorted per 
condition (e.g. GFP or Gi) or cluster (e.g. 1,2,3,4) to be further used in the analysis 

6. Scripts clusters_comparison_per_region.py and clusters_individual_statistics.py use the dictionaries from step (5) to compute zmaps comparison between GFP and Gi (stored in Zmaps_comparison)  and individual cluster statistics (stored in "Cluster Statistics" folder). 

7. Scripts diagram_plot.py and matrices_plot.py produce plots shown in Fig 3 D-F of the manuscript. Note Fig 3F left of the manuscript is truncated for the visualization purposes .

