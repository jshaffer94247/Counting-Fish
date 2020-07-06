# Leveraing Deep Learning to Facilitate Salmon Counting

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

Salmon life cycles follow a predictable pattern: hatch in fresh water, migrate to the ocean for the majority of their lives, and then migrate back to their original fresh water hatch sites before they spawn and then die. The time spent in fresh water and ocean salt water depends on the species.

Salmon populations in the waters of Puget Sound are estimated each year when a mature portion of the salmon migrate back from the ocean to fresh water to spawn. In many areas, this pathway is partially obstructed by boat locks (Seattle), or hydroelectric dams (Bonneville) and the salmon travel through carefully built fish ladders on this upstream journey. As they pass through the ladders, viewing windows allow them to be seen by both tourists and biologists, and human viewers are still the primary way to count the fish.

Once tallied, the estimated population for each species determines sport fishing limits such as the number of fish per day and the length of the fishing season. This data is also used to make decisions in the operation of salmon fisheries, commercial fishing, restaurants, and tourism. 

The salmon counting task is trivial when few are in the ladder; the task is far more difficult when many are returning at once. As a result, some locations estimate the full population by counting for a set period of time each day and comparing  to historical data. In other locations, 24/7 video recording enables biologists to review footage and tally the counts later; weekend tallies can take staff multiple days to catch up on counts.

## Data-Collection 

Over the course of 2 weeks in June 2020, an internet search found 168 usable images of fish traveling past viewing windows. Of these, the majority were taken by tourists and often feature the silhouettes of children in front of the glass. Images of official viewing windows were very difficult to find, in part because 1) they are probably not particularly interesting to most people and 2) for security reasons, the fish cam at the Bonneville Dam (Willamette Falls) has been disabled. 

With the use of image augmentation, the original collection of 168 images was expanded by including horizontal flip, random adjustments to exposure (+/- 25%), and random changes to rotation (+/- 15%). The final 504 images contained 725 annotated fish (averaging 4.3 per image), and included 2 null examples of viewing windows with no fish.

For image classification, images need to contain a limited number of objects (preferably just one) and a machine learning algorithm will attempt to name the object in the image. All that is needed is an image and a single label, e.g. "cat" or "dog".

Object detection refers to the case where there are multiple instances of an object or when there are a variety of other objects also in the image. In this situation, the image also needs to be labelled to show where each object is located. Most algorithms use a bounding box for this.

The original 168 fish images were manually labeled using the free tool "labelImg" (see https://pypi.org/project/labelImg/) to draw the bounding boxes. Free tools from roboflow.ai (see https://roboflow.ai/) were used to perform the image augmentation. Leveraging the roboflow tools provided several additional benefits: the bounding boxes were automatically adjusted for images that were randomly rotated, and the images and annotations could be quickly exported in multiple formats for use in a variety of models. 

## Deep-Learning-Models 

"You Only Look Once". YOLO is a popular object detection machine learning model introduced in 2015 by a group of researchers at the University of Washington. Rather than pass an image classifier multiple times over an image to see if there was, say, a dog at the upper left, or maybe at the upper right, this new approach replaced the final layers of an image classifier with additional convolutional layers that allowed it to find all instances in one pass. The immediate improvement in speed was a major leap forward for computer vision and object detection. Since the original paper, the model has been improved several times with Version 5 being released in June 2020.

Given the popularity, speed, and accuracy of YOLO, the YOLO v5 model flow available through roboflow.ai was an obvious choice. Earlier YOLO versions have keras and tensorflow implementations and can be run on a variety of hardware. At this time, only a PyTorch version of YOLO v5 has been built. This version leverages the computational speed and efficiency of a GPU for excellent results, and there are a number of examples available in blog posts and in github. For this project, the Google Colaboratory template from roboflow.ai was used. This template configures the environment and builds the model, so a simple customization consists of uploading a new training set and selecting the number of epochs for training. Once trained, the confidence threshold can be adjusted before making predictions.  

For this first model, it became apparent that labeling the fish by species was going to be highly problematic. First, identification is a challenge. Sport fishermen are discouraged from identifying fish by side view alone as this can be misleading; they are instead instructed to observe inside the mouth and to look at the color of the gum line. In cloudy, poorly lit conditions, other features such as silver highlights on the tail or where the black spots are located are very difficult to see. Second, training a model to recognize fish by species requires properly labeled images, and there were no fish experts working on this project. In lieu of counting by species, the project was scaled back to count them all as 'fish'.



## Conclusions

Based on the results from YOLO v5, salmon counting by object detection is definitely possible, and there also remain several challenges to be solved. These challenges include:

 - Viewing windows with excellent lighting are required
 - Viewing window height and width are not critical, but the depth needs to be carefully selected to reduce the number of fish that can obscure other fish
 - Correct species labels are required for training a model to separate coho from chinook



## References

### Salmon, salmon counting, and salmon fishing policies
 - https://www.nps.gov/olym/learn/nature/the-salmon-life-cycle.htm 
 - https://youtu.be/zoHpE5scs2I
 - http://www.fpc.org/currentdaily/HistFishTwo_7day-ytd_Adults.htm
 - https://wdfw.wa.gov/news/washingtons-salmon-seasons-tentatively-set-2020-21
 - http://pweb.crohms.org/tmt/documents/fpp/2020/final/FPP20_02_BON.pdf
 - https://idfg.idaho.gov/fish/chinook/dam-counts
 - http://www.eregulations.com/washington/fishing/salmon-identification/

### Machine learning

 - https://pypi.org/project/labelImg/
 - https://roboflow.ai/
 - https://arxiv.org/abs/1506.02640
 - https://towardsdatascience.com/how-to-train-a-custom-object-detection-model-with-yolo-v5-917e9ce13208
 
[Return to Table of Contents](#Contents)
