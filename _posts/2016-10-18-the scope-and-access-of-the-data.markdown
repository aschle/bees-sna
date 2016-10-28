---
title:  "The Scope of the Data set and Access"
date:   2016-10-18 15:00:00
categories: [a]
tags: [dataset, access, bb_binary]
refs: ["https://github.com/BioroboticsLab/bb_binary", "http://biorobotics.mi.fu-berlin.de/wordpress/", "http://www.fu-berlin.de/"]
---

The data set covers 64 days (about nine weeks) of honeycomb videos for August, September, and October of 2015. The two sides of the comb were filmed by two cameras each (left and right), resulting in four videos. Each video second per camera relates to three pictures, with corresponding bee detections. This adds up to 274 GB of detections stored in binary files in total, 4.5GB per day, and 200MB per hour.


The binary files can be accessed using the `bb_binary` library&nbsp;[1], written in python by the biorobotics lab&nbsp;[2] of Freie Universitäte Berlin&nbsp;[3].

![Folder structure of data set][structure]

[structure]: {{ site.url }}/images/structure.png



