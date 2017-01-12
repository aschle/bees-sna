#!/bin/bash
####################################
#
# Run pipleine_micro for different
# 2**X 
#
####################################

for i in 8 16 32 64 128 256 512
	do
		echo $i
		time python3 pipeline_micro.py "../00_Data/testset_2015_1h/2015082215/" 0.95 160 3 4 test-micro $i
	done
