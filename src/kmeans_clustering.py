import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_labeled(df, n_clusters):
    kmm = KMeans(n_clusters = n_clusters)
    kmm.fit(df)
    labeled_df = df.copy()
    labeled_df['cluster'] = kmm.labels_
    return labeled_df, kmm


if __name__=='__main__':
    # read df from pickle
    df_all = pickle.load(open('pkl/df_all.pkl', 'rb'))
    species_df = pickle.load(open('pkl/species_df.pkl', 'rb'))

    # Kmeans cluster on un-aggregated data
    all_labeled = kmeans_labeled(df_all, 22)

    # kmeans cluster on species-aggregated data
    sp_labeled, kmm = kmeans_labeled(species_df, 3)

    # scatter plots
    plt.scatter(sp_labeled['seasonal_temp'], sp_labeled['seasonal_rain'],
            c=sp_labeled['cluster'])
    plt.xlabel('seasonal temp [avg C]')
    plt.ylabel('seasonal rain [mm/year]')
    plt.colorbar()
    plt.show()

    # plot 2
    plt.scatter(sp_labeled['MeanAnnTemp'], sp_labeled['AvAnnRain_76_05'],
            c=sp_labeled['cluster'])
    plt.xlabel('mean annual temp [avg C]')
    plt.ylabel('avg annual rain [mm/year]')
    plt.colorbar()
    plt.show()

    # plot: cluster centers
    centers = kmm.cluster_centers_
    avgrain = centers[:,-2]
    avgtemp = centers[:,-1]

    plt.plot(avgtemp, avgrain, 'ok')
    plt.xlabel('mean annual temp [avg C]')
    plt.ylabel('avg annual rain [mm/year]')
    plt.show()
