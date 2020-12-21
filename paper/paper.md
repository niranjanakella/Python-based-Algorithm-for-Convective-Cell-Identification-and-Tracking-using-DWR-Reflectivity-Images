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
 - name: K V Subrahmanyam
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

# Acknowledgment

The current research has been supported with image data from MOSDAC, which helped in simulating and validating the developed algorithm at various levels.\\Authors are very greatly thankful to the Anaconda Inc. and OpenCV community for supporting the project through their valuable open-sourced content.

# References

- Y. Arnaud, M. Desbois and J. Maizi, “Automatic tracking and characterization of African convective systems on meteosat pictures”, Journal of Applied Meteorology, vol. 31, no. 5, (1992), pp. 443–453; DOI:https://doi.org/10.1175/1520-0450(1992)031<0443:ATACOA>2.0.CO;2

- Augustine, J.A., Howard, K.W., 1988. Mesoscale Convective Complexes over the United States during 1985. Mon. Wea. Rev. 116 (3), 685-701. DOI: doi.org/10.1175/1520-0493(1988)116%3C0685:MCCOTU%3E2.0.CO;2 
- Cancelada, M., Salio, P., Vila, D., Nesbitt, S.W., 315 Vidal, L., 2020. Backward Adaptive Brightness Temperature Threshold Technique (BAB3T): A Methodology to Determine Extreme Convective Initiation Regions Using Satellite Infrared Imagery. Remote Sens.12, 337. DOI: doi.org/10.3390/rs12020337
- Carvalho, L.M.V., Jones, C., 2001. A satellite method to identify structural properties of mesoscale convective systems based on maximum spatial correlation tracking technique (MASCOTTE). J. Appl. Meteor. 40, 1683–1701. DOI: https://doi.org/10.1175/1520-0450(2001)040<1683:ASMTIS>2.0.CO;2
- Chen, S.S., Houze, R.A., Mapes, B.E., 1996. Multiscale variability of deep convection in
relation to large‐scale circulation in TOGA COARE. J. Atmos. Sci. 53, 1380–1409. DOI: https://doi.org/10.1175/1520-0469(1996)053<1380:MVODCI>2.0.CO;2
- Cotton, W.R., Anthes, R.A., 1989. Storm and Cloud Dynamics. Academic Press, 880 pp; Google Books
- Fiolleau, T., Roca, R., 2013. An algorithm for the detection and tracking of tropical mesoscale convective systems using infrared images from geostationary satellite. IEEE Trans.Geosci. Remote. Sens. 51(7), 4302-4315
- Gray, W.M., 1998. The formation of tropical cyclones. Meteorol. Atmos. Phys. 67, 37–69
- Houze, R.A., 1982. Cloud clusters and large-scale vertical motions in the tropics. J. Meteorol.
332 Soc. Jpn. 60, 396–410Houze, R.A., 2004. Mesoscale convective systems. Rev. Geophys. 42(4)
- Kober, K., Tafferner, A., 2009. Tracking and nowcasting of convective cells using remote sensing data from radar and satellite. Meteorol. Zeitsch. 18, 75-84. DOI: doi.org/10.1127/0941-2948/2009/359Lakshmanan, V., Hondl, K., Rabin, R., 2009. An efficient, general-purpose technique for identifying storm cells in geospatial images. J. Atmos. Oceanic. Technol. 26(3), 523-537
- Lang, P., 2001. Cell tracking and warning indicators derived from operational radar products. Proceedings of the 30th International Conference on Radar Meteorology, Munich, Germany, 245–247
- Mecklenburg, S., Joss, J., Schmidt, S.W., 2000. Improving the nowcasting of precipitation in an Alpine region with enhanced radar echo tracking algorithm. J. Hydrol. 239, 46–68
- Moncrieff, M.W., 2010. The multiscale organization of moist convection and the intersection of weather and climate. Why Does ClimateVary? American Geophysical Union, Climate Dynamics. DOI: https://doi.org/10.1029/2008GM000838 Washington, DC
- Nunez Ocasio, K.M., Evans, J.L., Young, G.S., 2020. Tracking Mesoscale Convective Systems that are Potential Candidates for Tropical Cyclogenesis. Mon. Wea. Rev. (148(2), 655–669. doi.org/10.1175/MWR-D-19-0070.1
- Subrahmanyam, K.V., Kumar, K.K., 2013. CloudSat observations of cloud-type distribution over the Indian summer monsoon region. Ann. Geophys. 31, 1155-1162. doi: 10.5194/angeo-31-1155-2013.
- Subrahmanyam, K.V., Kumar, K.K, Reddy, N., 2020. New insights into the convective system characteristics over the Indian summer monsoon region using space based passive and active remote sensing techniques. IETE Technical Revie. 37(2), 211-219.
DOI: doi.org/10.1080/02564602.2019.1593890.
- Teng, H.F., Lee, C.S., Hsu, H.H., 2014. Influence of ENSO on formation of tropical cloud clusters and their development into tropical cyclones in the western North Pacific. Geophys. Res. Lett. 41, 9120–912
- Velasco, I., Fritsch, J.M., 1987. Mesoscale convective complexes in the Americas. J. Geophys.Res. 92, 9591–9613
- Williams, M., Houze, R.A., 1987. Satellite-observed characteristics of winter monsoon cloud clusters. Mon. Wea. Rev. 115(2), 505–519.
- Xu, W.E., Zipser, J., 2012. Properties of deep convection in tropical continental, monsoon, and oceanic rainfall regimes. Geophys. Res. Lett. 39(7). DOI: doi.org/10.1029/2012GL051242
- Zinner, T., Mannstein, H., Tafferner, A., 2008. Cb-tram: Tracking and monitoring severe convection from onset over rapid development to mature phase using multi-channel meteosa-8 seviri data. Meteor. Atmos. Phys. 101(3-4), 191–210