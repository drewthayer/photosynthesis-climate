import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def scaled_df(df):
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_df = scaler.transform(df)
    return scaled_df

def ranked_zip(labels, values, abs_vals=True):
    if abs_vals:
        zipped = zip(labels, abs(values))
    else:
        zipped = zip(labels, values)
    return sorted(zipped, key=lambda x: x[1], reverse=True)

def ranked_components_plot(labels, values, title, abs_vals=True):
    if abs_vals:
        zipped = zip(labels, abs(values))
    else:
        zipped = zip(labels, values)
    srted = sorted(zipped, key=lambda x: x[1], reverse=True)

    # plot
    arr = np.array(srted)
    idx = np.arange(arr.shape[0])
    plt.barh(idx, arr[:,1])
    plt.yticks(idx, arr[:,0])
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    # read df from pickle
    species_df = pickle.load(open('pkl/species_df.pkl', 'rb'))

    # scale data
    scaled = scaled_df(species_df)

    # pca
    n_components = 2
    pca = PCA(n_components)
    pca.fit(scaled)
    df_new = pca.transform(scaled)
    comps = pca.components_
    print('PCA: {} components'.format(n_components))

    # explained variance ratio
    ratio = pca.explained_variance_ratio_
    print('explained variance ratio = {:0.2f} : {:0.2f}'.format(ratio[0], ratio[1]))

    # components sorted by feature loadings
    pc1_ranked = ranked_zip(species_df.columns, comps[0,:], abs_vals=True)
    pc2_ranked = ranked_zip(species_df.columns, comps[1,:], abs_vals=True)

    # component loading plots
    pcs = {'pc1': comps[0,:],
            'pc2': comps[1,:]}
    for pc in pcs:
        ranked_components_plot(species_df.columns, pcs[pc], pc, abs_vals=True)
    #ranked_components_plot(species_df.columns, comps[0,:], title=pc[0], abs_vals=True)
    #ranked_components_plot(species_df.columns, comps[1,:], abs_vals=True)
