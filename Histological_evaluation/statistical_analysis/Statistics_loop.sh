#!/bin/bash

# Define the base directory containing the Cluster_*_data directories
BASE_DIR="/mnt/d/Parcellation/Github/Histological_evaluation/statistical_analysis"

# Loop over clusters 1 to 4
for i in 1 2 3 4; do
    DATA_DIR="${BASE_DIR}/Cluster_${i}_data"
    OUTPUT_FILE="voxel_counts_${i}.csv"
    
    # Clear the output file if it already exists
    > "$OUTPUT_FILE"
    
    echo "Processing directory: $DATA_DIR"
    
    for file in "$DATA_DIR"/*.nii.gz; do
        # Extract the voxel count for value 1
        voxel_count=$(fslstats "$file" -l 0.5 -u 1.5 -V | awk '{print $1}')
        
        # Extract the total number of non-zero voxels for a double-check, not encluded in Analysis.xlsx
        total_voxels=$(fslstats "$file" -V | awk '{print $1}')
        
        # Get the filename without the path
        filename=$(basename "$file")
        
        # Save the result to the CSV file and print a processing message
        echo "$filename, $voxel_count, $total_voxels" >> "$OUTPUT_FILE"
        echo "Processed: $filename -> Voxels with value 1: $voxel_count, Total Non-Zero voxels: $total_voxels"
    done
done

