#!/bin/bash


for i in {90..100..1}
    do
       python3 pipeline_frames_parallel.py \
       /mnt/data2016/ \
       2016-07-21T15:00:00Z \
       60 \
       $i \
       212 \
       3 \
       8 \
       2016-07-21T15:00:00Z_1h_allCams-$i \
       2016 \
       2 \
       10
done