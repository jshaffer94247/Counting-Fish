# Leveraging Deep Learning to Facilitate Salmon Counting

### Contents
- [Executive Summary](#Executive-Summary)
- [Introduction](#Introduction)
- [Data Collection](#Data-Collection)
- [Deep Learning Models](#Deep-Learning-Models)
- [Conclusions](#Conclusions)
- [References](#References)

## Executive Summary

In the Pacific Northwest, salmon are a key component of both the commerce and the ecosystem, and population estimates are key factors in many decisions. Current methods require trained biologists to observe the fish passing a viewing window and manually record the fish count. 

This project explored the possibility of using machine learning such as object detection and classification to support these efforts, possibly enabling the collection of data in more locations and over longer time periods.

Custom trained models (e.g. YOLO v5) using images from fish ladders proved that accurate fish detection is promising but non-trivial, counting fish in a still image does not solve the problem of counting fish in video, and that classifying fish by species requires excellent viewing conditions.

## Introduction

Salmon life cycles follow a predictable pattern: hatch in fresh water, migrate to the ocean for the majority of their lives, and then migrate back to their original fresh water homes before they spawn and then die. The time spent in fresh water and ocean salt water depend on the species.

Salmon populations in the waters of Puget Sound are estimated each year when a mature portion of the salmon migrate back from the ocean to fresh water to spawn. In many areas, this pathway is partially obstructed by boat locks (Seattle), or hydroelectric dams (Bonneville) and the salmon travel through carefully built fish ladders on this upstream journey. As they pass through the ladders, viewing windows allow them to be seen by both tourists and biologists, and human viewers are still the primary way to count the fish.

Once tallied, the estimated population for each species determines sport fishing limits such as the number of fish per day and the length of the fishing season. This data is also used to make decisions in the operation of salmon fisheries, commercial fishing, restaurants, and tourism. 

The salmon counting task is trivial when few are in the ladder; the task is far more difficult when many are returning at once. As a result, some locations estimate the full population by counting for a set period of time each day and comparing  to historical data. In other locations, 24/7 video recording enables biologists to review footage and tally the counts later; weekend tallies can take staff multiple days to catch up on counts.


[Return to Table of Contents](#Contents)



## Data-Collection 

Over the course of 2 weeks in June 2020, an internet search found only 168 usable images of fish travelling past viewing windows. Of these, the majority were taken by tourists and often feature the silhouettes of children in front of the glass. Images of official viewing windows were very difficult to find for 2 reasons: 1) they are probably not particularly interesting to most people and 2) for security reasons, the fish cam at the Bonneville Dam (Willamette Falls) has been disabled. 

With the use of image augmentation, the original collection of 168 images was expanded by including horizontal flip, random adjustments to exposure (+/- 25%), and random changes to rotation (+/- 15%). The final 504 images contained 725 annotated fish (averaging 4.3 per image), and included 2 null examples of viewing windows with no fish.


## Deep-Learning-Models 



## Conclusions





## References


### Salmon, salmon counting, and salmon fishing policies
 - https://www.nps.gov/olym/learn/nature/the-salmon-life-cycle.htm 
 - https://youtu.be/zoHpE5scs2I
 - http://www.fpc.org/currentdaily/HistFishTwo_7day-ytd_Adults.htm
 - https://wdfw.wa.gov/news/washingtons-salmon-seasons-tentatively-set-2020-21
 - http://pweb.crohms.org/tmt/documents/fpp/2020/final/FPP20_02_BON.pdf
 - https://idfg.idaho.gov/fish/chinook/dam-counts

### Machine learning



[Return to Table of Contents](#Contents)