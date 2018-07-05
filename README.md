## project: photosynthesis-climate signal: clustering and dimensionality reduction
 This a side project to provide support for my wife's post-doctoral research, in collaboration with Brown University (RI, USA) and James Cook University (Queensland, Australia). The researchers study variations of CAM photosynthesis within succulent plants in Australia, and one hypothesis they're working with suggests that particular types of CAM photosynthesis should vary according to climatic signals, largely rainfall and temperature.

 Briefly, CAM photosynthesis is an adaptation from the globally common C3 photosynthesis which helps plants adapt to hot and/or dry environments. CAM's main mechanisms include closing stomata (respiration holes in leaves) during the day to avoid water loss, and storing photosynthetic energy during the day (as malic acid) for use during respiration (sugar production) at night. Cam plants are characterized by specific and unique chemical and physical adaptations which make these processes possible.

 There are three variants of CAM photosynthesis:
  - full CAM: plants use CAM all the time
  - facultative CAM: plants possess chemical and physical adaptations for CAM, but only use it some parts of the year
  - low-level cam: plants use a low level of CAM most of the time

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

### Principal Component Analysis (PCA): which general climatic factors most explain variation in species?

From the __principal component loadings__ we can see that features related to rainfall explain the most variance in the data, followed by features related to sun hours and temperature.

<img alt="pca loadings" src="/figs/pc1_pc2_components.png" size="300">
