import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_labeled(df, n_clusters):
    ''' fit kmeans clustering to dataframe, return dataframe and
    fitted estimator'''
    kmm = KMeans(n_clusters = n_clusters)
    kmm.fit(df)
    labeled_df = df.copy()
    labeled_df['cluster'] = kmm.labels_
    return labeled_df, kmm


if __name__=='__main__':
    # read df from pickle
    df_all = pickle.load(open('pkl/df_all.pkl', 'rb'))
    species_df = pickle.load(open('pkl/species_df.pkl', 'rb'))
    phenotype = species_df.pop('Phenotype')

    # Kmeans cluster on un-aggregated data
    all_labeled = kmeans_labeled(df_all, 22)

    # kmeans cluster on species-aggregated data
    sp_labeled, kmm = kmeans_labeled(species_df, 3)

    # scatter plot for seasonal precip and temp, labeled by kmeans cluster
    fig = plt.subplots(figsize=(7,5.5))
    plt.scatter(sp_labeled['seasonal_temp'], sp_labeled['seasonal_rain'],
            c=sp_labeled['cluster'])
    plt.xlabel('mean growing season temp [avg C]')
    plt.ylabel('mean growing season rain [mm/year]')
    cbar = plt.colorbar(ticks=[0,1,2])
    cbar.ax.set_yticklabels(['group 1', 'group 2', 'group 3'])
    #plt.colorbar()
    plt.tight_layout()
    plt.show()


    # again, labeled by pre-labeled phenotype
    fig = plt.subplots(figsize=(7,5.5))
    plt.scatter(sp_labeled['seasonal_temp'], sp_labeled['seasonal_rain'],
            c=phenotype)
    plt.xlabel('mean growing season temp [avg C]')
    plt.ylabel('mean growing season rain [mm/year]')
    cbar = plt.colorbar(ticks=[0,1,2])
    cbar.ax.set_yticklabels(['phenotype 1', 'phenotype 2', 'phenotype 3'])
    #plt.colorbar()
    plt.tight_layout()
    plt.show()

    # plot 2
    fig = plt.subplots()
    plt.scatter(sp_labeled['MeanAnnTemp'], sp_labeled['AvAnnRain_76_05'],
            c=sp_labeled['cluster'])
    plt.xlabel('mean annual temp [avg C]')
    plt.ylabel('mean annual rain [mm/year]')
    #cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    #cbar.ax.set_yticklabels(['< -1', '0', '> 1'])
    cbar = plt.colorbar(ticks=[0,1,2])
    cbar.ax.set_yticklabels(['type1', 'type2', 'type3'])
    plt.tight_layout()
    plt.show()

    # plot: cluster centers
    centers = kmm.cluster_centers_
    avgrain = centers[:,-2]
    avgtemp = centers[:,-1]

    plt.plot(avgtemp, avgrain, 'ok')
    plt.xlabel('mean annual temp [avg C]')
    plt.ylabel('avg annual rain [mm/year]')
    plt.show()

    from matplotlib.patches import Ellipse

    def plot_ellipse(mean, var, ec='k', alpha=1):
        evals, evecs = np.linalg.eig(var)
        ang = np.degrees(np.arctan2(*evecs[1]))
        ell = Ellipse(mean, *np.abs(evals), angle=ang, fc='None', ec=ec, alpha=alpha)
        plt.gca().add_artist(ell)

    
