#!/bin/bash
####################################
#
# Run pipeline with different params
#
####################################

# interaction length
for i in 3, 6, 9, 12, 15
	do
		for j in 100, 115, 130, 145, 160
			do
				echo $i $j
				time python3 pipeline.py "/storage/mi/aschle/1day/20150822/" 0.95 $j $i 8
			done
	done