#!/bin/bash 

#declare -a corr_types=("pfc" "all" "nopfc")
#declare -a data_batches=(20 40 60 80 100)

declare -a corr_types=("pfc" "all" "nopfc")
declare -a data_batches=(20 40 60 80)

for data in "${data_batches[@]}"
	do
	for types in "${corr_types[@]}"
		do
			#echo  Condition $types $data is proceeded
			bsub -J "$types"_"$data" -o "log"_"$types"_"$data" -R "rusage[mem=50000]" -W 12:00 python Euler_correlation_combined.py $types $data
		done
	done
