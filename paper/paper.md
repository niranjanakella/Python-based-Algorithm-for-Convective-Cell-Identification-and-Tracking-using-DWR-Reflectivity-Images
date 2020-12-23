---
title: 'Python-based Algorithm for Convective Cell Identification and Tracking using DWR Reflectivity Images'
tags:
  - Doppler Weather Radar
  - Computer Vision
  - LSTM Nerual Networks
  - OCR
  - Optical Character Recognition
  - Python Algorithm
  - Convective Cell Tracking
authors:
 - name: Akella Niranjan
   orcid: 0000-0002-0625-236X
   affiliation: 1
 - name: Kandula V Subrahmanyam
   orcid: 0000-0003-2987-1232
   affiliation: 2
 - name: S V Ranganayakulu
   orcid: 0000-0003-1070-5103
   affiliation: 3
affiliations:
 - name: 3rd Year, Electronics & Communication Engineering, Guru Nanak Institutions Technical Campus, Ibrahimpatnam, Telangana, India
   index: 1
 - name: Scientist/Engineer, ‘SE’, Space Physics Laboratory, Vikram Sarabhai Space Centre, ISRO, Trivandrum, India
   index: 2
 - name: Dean Research & Development, Guru Nanak Institutions Technical Campus, Ibrahimpatnam, Telangana, India
   index: 3
date: 21 December 2020
bibliography: paper.bib
---

# Summary
Owing to its importance in the hydrological cycle, Convective Cells play a major role in the Earth's atmosphere.They result in a very large proportion of precipitation at various altitudes, [@Cotton,@Moncrieff,@Chen]. These Convective Cells evolve through time forming dense storms which play a major role in the atmospheric exchange between stratosphere & troposphere, [@Xu]. The identification and tracking of these Convective Cells through out their lifetime are of great significance in the weather and climate system. The present study developed an autonomous Python-based algorithm for the identification and tracking of Convective Cells using Doppler Weather Radar (DWR) reflectivity images and the preliminary results are discussed. The proposed algorithm implements Deep Neural Network-based Computer Vision (CV) approach for the identification and tracking of Convective Cells using Optical Character Recognition (OCR) engine "Tesseract". The Tesseract-engine is an unsupervised Neural Network module based on Long Short Term Memory (LSTM) which analyses the input image in the form of a dimensional pixel array, which is further processed through each layer of the Neural Network to predict the desired output based on the trained weights and also implementing connected component analysis. The algorithm runs through the DWR reflectivity image pixel values and recognizes the pixels intensities which are then segregated to recognize the convective cells along with other estimated cell properties such as the Centroid of Convective Cell, the area and the distance covered by the Convective Cell with the  direction of the dense formation from the radar centre which is considered to be the origin. The developed algorithm takes in a series of consecutive DWR images in a folder as an argument and then performs the necessary computations needed to identify and track the Convective Cells at each given time-stamp and finally plots the path followed by each Convective Cell throughout its lifetime. The performance of the proposed algorithm has been tested on various convective storms and it could successfully identify and track the Convective Cells along with their physical properties. Currently, the algorithm takes only reflectivity images as single input parameter. Future work with the proposed algorithm includes the identification of convective systems that are developed into tropical cyclones and estimating the rainfall contribution from convective cells.


# Statement of Need
The proposed Python algorithm has a Deep learning-based approach towards the identification and tracking of Convective Cells through various time stamps directly from Doppler Weather Radar reflectivity images which exponentially decreses the time complexity and space complexity towards aquiring desired outputs. Since the proposed algorithm performs all the computations directly on the images rather than binary data from the Radar itself, it can process the desired outputs much faster compared to other algorithms which process the binary data from the Radar. 

# Acknowledgment

The current research has been supported with image data from MOSDAC, which helped in simulating and validating the developed algorithm at various levels. 

Authors are very greatly thankful to the Anaconda Inc. and OpenCV community for supporting the project through their valuable open-sourced content.

# References