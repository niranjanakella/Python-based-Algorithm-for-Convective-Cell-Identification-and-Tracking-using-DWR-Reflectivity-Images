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
 - name: Space Physics Laboratory, Vikram Sarabhai Space Centre, ISRO, Trivandrum, India
   index: 2
 - name: Dean Research & Development, Guru Nanak Institutions Technical Campus, Ibrahimpatnam, Telangana, India
   index: 3
date: 21 December 2020
bibliography: paper.bib
---

# Summary
Owing to its importance in the hydrological cycle, Convective Cells play a major role in the Earth's atmosphere.They result in a very large proportion of precipitation at various altitudes. The identification and tracking of these Convective Cells through out their lifetime are of great significance in the weather and climate system. The present study developed an autonomous Python-based algorithm for the identification and tracking of Convective Cells using Doppler Weather Radar (DWR) reflectivity images and the preliminary results are discussed. The proposed algorithm implements Deep Neural Network-based Computer Vision (CV) approach for the identification and tracking of Convective Cells using Optical Character Recognition (OCR) engine "Tesseract". The Tesseract-engine is an unsupervised Neural Network module based on Long Short Term Memory (LSTM) which analyses the input image in the form of a dimensional pixel array, which is further processed through each layer of the Neural Network to predict the desired output based on the trained weights and also implementing connected component analysis. The algorithm runs through the DWR reflectivity image pixel values and recognizes the pixels intensities which are then segregated to recognize the convective cells along with other estimated cell properties such as the Centroid of Convective Cell, the area and the distance covered by the Convective Cell with the  direction of the dense formation from the radar centre which is considered to be the origin. The performance of the proposed algorithm has been tested on various convective storms and it could successfully identify and track the Convective Cells along with their physical properties. Currently, the algorithm takes only reflectivity images as single input parameter. Future work with the proposed algorithm includes the identification of convective systems that are developed into tropical cyclones and estimating the rainfall contribution from convective cells.

# Statement of Need
The proposed Python algorithm has a Deep learning-based approach towards the identification and tracking of Convective Cells through various time stamps directly from Doppler Weather Radar reflectivity images which exponentially decreses the time complexity and space complexity towards aquiring desired outputs. Since the proposed algorithm performs all the computations directly on the images rather than binary data from the Radar itself, it can process the desired outputs much faster compared to other algorithms which process the binary data from the Radar. 

# Acknowledgment

The current research has been supported with image data from MOSDAC, which helped in simulating and validating the developed algorithm at various levels.\\Authors are very greatly thankful to the Anaconda Inc. and OpenCV community for supporting the project through their valuable open-sourced content.

# References

Y.  Arnaud,  M.  Desbois  and  J.  Maizi;1992 V31(443-453); Automatic  tracking  and  char-acterization  of  African  convective  systems  on  meteosat  pictures; Journal of Applied Meteorology; DOI: 10.1175/1520-0450(1992)031<0443:ATACOA>2.0.CO;2

Augustine, J.A., Howard, K.W.; 1988 V116 I3; Mesoscale Convective Complexes over the United States during 1985; Journal of Applied Meteorology; DOI: 10.1175/1520-0493(1988)116<0685:MCCOTU>2.0.CO;2

Cancelada, M., Salio, P., Vila, D., Nesbitt, S.W., 315 Vidal, L; 2020 V12 I2; Backward Adaptive Brightness Temperature Threshold Technique (BAB3T): A Methodology to Determine Extreme Convective Initiation Regions Using Satellite Infrared Imagery; Journal of Remote Sensing, MDPI; DOI: 10.3390/rs12020337

Carvalho, L.M.V., Jones, C; 2001 V40 I10; Multiscale variability of deep convection in relation to large‐scale circulation in TOGA COARE; Journal of Atmospheric Sciences; DOI: 10.1175/1520-0469(1996)053<1380:MVODCI>2.0.CO;2

Chen, S.S., Houze, R.A., Mapes, B.E; 1996 V53 I10; Multiscale variability of deep convection in relation to large‐scale circulation in TOGA COARE; Journal of Atmospheric Sciences; DOI: 10.1175/1520-0469(1996)053<1380:MVODCI>2.0.CO;2

Cotton, W.R., Anthes, R.A.; 2010 V99 I2; Book Title: Storm and Cloud Dynamics; ScienceDirect; https://www.sciencedirect.com/bookseries/international-geophysics/vol/99/suppl/C

T. Fiolleau and R. Roca; 2013 V51 I7; An Algorithm for the Detection and Tracking of Tropical Mesoscale Convective Systems Using Infrared Images From Geostationary Satellite; IEEE Transactions on Geoscience and Remote Sensing; DOI: 10.1109/TGRS.2012.2227762

Gray, W.M.; 1998 V67 I1; The formation of tropical cyclones. Meteorol; Meteorology and Atmospheric Physics; DOI: 10.1007/BF01277501

Robert A, Houze, Jr.; 1982 V60 I1; Cloud clusters and large-scale vertical motions in the tropics; Journal of the Meteorological Society of Japan. Ser. II; DOI: 10.2151/jmsj1965.60.1_396

Kober, Kirstin and Tafferner, Arnold; 2009 V18(75-84); Tracking and nowcasting of convective cells using remote sensing data from radar and satellite; Meteorologische Zeitschrift; DOI: 10.1127/0941-2948/2009/359

Lang, P; 2001; Cell tracking and warning indicators derived from operational radar products; International Conference on Radar Meteorology, Munich, Germany; URL: https://ams.confex.com/ams/pdfpapers/21678.pdf

S Mecklenburg and J Joss and W Schmid; 2000 V239; Improving the nowcasting of precipitation in an Alpine region with an enhanced radar echo tracking algorithm; Journal of Hydrology; DOI: 10.1016/S0022-1694(00)00352-8

Moncrieff, M.W.; 2010 V189; The multiscale organization of moist convection and the intersection of weather and climate. Why Does ClimateVary? American Geophysical Union, Climate Dynamics; Geophysical Monograph Series; DOI: 10.1029/2008GM000838

Nunez Ocasio, K.M., Evans, J.L., Young, G.S.; 2020 V148 I2 Tracking Mesoscale Convective Systems that are Potential Candidates for Tropical Cyclogenesis; Monthly Weather Review; DOI: 10.1175/MWR-D-19-0070.1

Subrahmanyam, K.V., Kumar, K.K.; 2013; CloudSat observations of cloud-type distribution over the Indian summer monsoon region; Annales Geophysicae; DOI: 10.5194/angeo-31-1155-2013

Subrahmanyam, K.V., Kumar, K.K, Reddy, N.; 2020 V37 I2; New insights into the convective system characteristics over the Indian summer monsoon region using space based passive and active remote sensing techniques; IETE Technical Review; DOI: 10.1080/02564602.2019.1593890

Teng, H.F., Lee, C.S., Hsu, H.H.; 2014 V41 I24; Influence of ENSO on formation of tropical cloud clusters and their development into tropical cyclones in the western North Pacific; Geophysical Research Letters; DOI: 10.1002/2014GL061823

Velasco, I., Fritsch, J.M.; 1987 V92; Mesoscale convective complexes in the Americas; Journal of Geophysical Research; DOI: 10.1029/JD092iD08p09591

Williams, M., Houze, R.A.; 1987 V115 I2; Satellite-observed characteristics of winter monsoon cloud clusters; Monthly Weather Review; DOI: 10.1175/1520-0493(1987)115<0505:SOCOWM>2.0.CO;2

Xu, W.E., Zipser, J.; 2012 V39 I7; Properties of deep convection in tropical continental, monsoon, and oceanic rainfall regimes; Geophysical Research Letters; DOI: 10.1029/2012GL051242

Zinner, T., Mannstein, H., Tafferner, A.; 2008 V101 I3; Cb-tram: Tracking and monitoring severe convection from onset over rapid development to mature phase using multi-channel meteosa-8 seviri data. Meteor; Meteorology and Atmospheric Physics; DOI: 10.1007/s00703-008-0290-y