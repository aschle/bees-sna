---
title:  "The Dataset"
date:   2016-10-16 15:00:00
categories: [a]
tags: [dataset]
refs: ["https://github.com/BioroboticsLab/bb_binary/blob/master/bb_binary/bb_binary_schema.capnp#L26"]
---

The starting point of my thesis is a spatial-temporal dataset consisting of tracked and detected honeybees during the summer season in 2016.

![bees with 12-bit markers][markers]

The basis for the dataset is nine weeks of video data capturing honeybees moving across the honeycomb. Each individual of the colony, including about 3000 bees, were tagged with 12-bit markers. Each video file consists of 1024 frames, which corresponds to an approximately 6-minute video. One second video consists of three frames. Per frame or timestamp bees are detected with the following parameters:

**Detection scheme [1]:**

* x coordinate in px
* y coordinate in px
* direction of head (rotation in z-plane) in radian
* radius of the ID tag
* decoded 12-bit id with probabilities discretised to 0-255


[markers]: {{ site.url }}/images/markers.png