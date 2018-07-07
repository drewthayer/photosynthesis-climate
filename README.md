## Photosynthesis-climate signal: clustering and dimensionality reduction
 This a side project to support my wife's post-doctoral research. She collaborates with researchers at Brown University (RI, USA) and James Cook University (Queensland, Australia) to study variations of CAM photosynthesis within succulent plants in Australia.

  One hypothesis they're working with suggests that particular types of CAM photosynthesis should vary according to climatic signals, likely rainfall and temperature.

 _Briefly: CAM photosynthesis has evolved hundreds of times from the globally common, C3 photosynthetic pathway and it helps plants live in hot and/or dry environments. CAM is a water-use efficient form of photosynthesis that is characterized by the plants capacity to fix CO2 in the dark and to store the carbon overnight in malic acid until daylight resumes. Cam plants are characterized by specific and unique biochemical, anatomical, and physiological adaptations which make these processes possible._

 There are three variants of CAM photosynthesis:
  - __strong CAM:__ plants use CAM all the time
  - __facultative CAM:__ plants primarily use C3 photosynthesis but under stress (eg., drought, salinity) switch to CAM. When the stress is relieved the plants revert back to C3 photosynthesis.
  - __low-level cam:__ plants that use a little CAM all of the time but fix most of their CO2 using C3 photosynthesis.

### goals:
 This project has two goals:
 
   1. to use clustering and dimensionality reduction to find groups of CAM-evolved species that live within similar climate spaces.  
   2. to see if certain phenotypes are selected for in certain environments -- basically asking, is there an adaptive advantage for plants to use these C3+CAM phenotypes in certain environments?


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

## Principal Component Analysis (PCA): which general climatic factors most explain variation in species?
The first and second principal components explain 60% and 34% of the variance in the data, respectively.

From the __principal component loadings__ we can see that features related to rainfall explain the most variance in the data, followed by features related to sun hours and temperature.

<img alt="pca loadings" src="/figs/pc1_pc2_components.png" width="800">

__principal component reduction plot:__

<img alt="pca xplot" src="/figs/pca_crossplot.png" width="300">

Although the first two principal components explain much of the variation in the data, they do not by themselves clearly separate the labeled phenotypes.

## K-Means Clustering analysis: finding species that live in similar climatic conditions

Clustered into 3 groups, the data group as follows:

<img alt="cluster" src="/figs/temp_precip_growseason.png" width="300">

Labeled by phenotype:

<img alt="cluster" src="/figs/temp_precip_growseason_pheno.png" width="320">

These cluster labels do not correlate well with phenotype (only match in 13% of species)
