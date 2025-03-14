#!/bin/bash 

declare -a corr_types=("pfc-pfc" "pfc-nopfc" "pfc-brain")

for types in "${corr_types[@]}"
	do
		for animal_number in $(seq 0 0) # substitute second 0 to 99 to run the whole database (should be stored in Data folder)
		do 
			echo "$types"_"$animal_number"
			python correlation_individual.py $types $animal_number
			# You can also compute the correlations using cluster computing. Below is an example with IBM LSF.
			#bsub -J "$types"_"$animal_number" -o "log"_"$types"_"$animal_number" -R "rusage[mem=15000]" -W 4:00 python correlation_individual.py $types $animal_number
			
		done
	done


