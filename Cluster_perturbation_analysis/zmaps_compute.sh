##!/bin/bash

# Masks for the clusters
mask="/mnt/d/Parcellation/2024/Scripts_2024/Cluster_masks"

# Used data
data="/mnt/d/Parcellation/2024/Scripts_2024/Data/Gi_masked"
#data="/mnt/d/Parcellation/2024/Scripts_2024/Data/GFP_masked"


# directories for saving the resultsb
timeseries="/mnt/d/Parcellation/2024/Scripts_2024/timeseries"
zstats="/mnt/d/Parcellation/2024/Scripts_2024/zstats"

cluster_number=4

# clearing the working directories
rm -rf $zstats
rm -rf $timeseries


################
#computing average timeseries from regions defined by individual masks
###############

for d in "$data"/*
	do
        filename=$(basename "$d")
        if [[ $filename =~ ag([a-zA-Z0-9]+)_ ]]; then
            data_number="${BASH_REMATCH[1]}"
        else
            echo "No match found in filename: $filename"
            continue  # Skip this iteration if data_number is not found
        fi
		for c in `seq 1 $cluster_number` #adjust to the cluster number
			do 
			echo "timeseries for $data_number and cluster $c"
			
			fslmeants -i "$d" -o "$timeseries"/"$c"/"$data_number"_"$c".txt -m "$mask"/"kmeans-$cluster_number-pfc_all-$c.nii.gz"
			
		done
	done

################
# computing individual zstats
################

for d in "$data"/*
	do
	#fslcpgeom $underlay $d
        filename=$(basename "$d")
        if [[ $filename =~ ag([a-zA-Z0-9]+)_ ]]; then
            data_number="${BASH_REMATCH[1]}"
        else
            echo "No match found in filename: $filename"
            continue  # Skip this iteration if data_number is not found
        fi
	
		for c in `seq 1 $cluster_number` 		
		    do 
		    echo "zstats for $data_number and cluster $c"
		#bsub -J "$data_number"_"$c"_"z" -w "done($last_job_name)" -R "rusage[mem=4608]" -o "$log_zstats"/"$data_number"_"$c"_"fsl" fsl_glm -i "$d" -d "$timeseries"/"$c"/"$data_number"_"$c".txt -m "$brain_mask" --demean --out_z="$zstats"/"$c"/"$data_number"_"$c"_"z".nii.gz # potential example for a run in the cluster, not used for this data.

		    fsl_glm -i "$d" -d "$timeseries"/"$c"/"$data_number"_"$c".txt  --demean --out_z="$zstats"/"$c"/"$data_number"_"$c"_"z".nii.gz

		    done
	done


