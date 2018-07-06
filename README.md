## Photosynthesis-climate signal: clustering and dimensionality reduction
 This a side project to support my wife's post-doctoral research. She collaborates with researcheres at Brown University (RI, USA) and James Cook University (Queensland, Australia) to study variations of CAM photosynthesis within succulent plants in Australia.

  One hypothesis they're working with suggests that particular types of CAM photosynthesis should _vary according to climatic signals_, likely rainfall and temperature.

 _Briefly, CAM photosynthesis is an adaptation from the globally common C3 photosynthesis which helps plants adapt to hot and/or dry environments. CAM's main mechanisms include closing stomata (respiration holes in leaves) during the day to avoid water loss, and storing photosynthetic energy during the day (as malic acid) for use during respiration (sugar production) at night. Cam plants are characterized by specific and unique chemical and physical adaptations which make these processes possible._

 There are three variants of CAM photosynthesis:
  - __full CAM:__ plants use CAM all the time
  - __facultative CAM:__ plants possess chemical and physical adaptations for CAM, but only use it some parts of the year
  - __low-level cam:__ plants use a low level of CAM most of the time

### purpose:
 The purpose of this project is to use clustering and dimensionality reduction to find groups of CAM-evolved species that live within similar climate spaces.

### data:
  - these data are based on individual species samples (collections) collected in Australia.
  - climate data has been collected from geo-spatial databases for each collection
      - _climate variables:_
      - RainDays_10mm
      - RainDays_5mm
      - EvapTran_mm
      - RelHumid_9am
      - MaxAnnTemp
      - AvgSunHrs
      - seasonal_rain
      - seasonal_temp
      - AvAnnRain_76_05
      - MeanAnnTemp
  - this dataset includes 8402 collections which represent 22 species
  - each species has been labeled with a photosynthetic phenotype

### Principal Component Analysis (PCA): which general climatic factors most explain variation in species?
The first and second principal components explain 60% and 34% of the variance in the data, respectively.

From the __principal component loadings__ we can see that features related to rainfall explain the most variance in the data, followed by features related to sun hours and temperature.

<img alt="pca loadings" src="/figs/pc1_pc2_components.png" size="300">

__principal component reduction plot:__

<img alt="pca xplot" src="/figs/pca_crossplot.png" size="200">

Although the first two principal components explain much of the variation in the data, they do not by themselves clearly separate the labeled phenotypes.

### K-Means Clustering analysis: finding species that live in similar climatic conditions

Clustered into 3 groups, the data group as follows:

<img alt="cluster" src="/figs/temp_precip_growseason.png" size="200">

Labeled by phenotype:

<img alt="cluster" src="/figs/temp_precip_growseason_pheno.png" size="200">

These cluster labels do not correlate well with phenotype (only match in 13% of species)
