import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pdb

# imports for standard font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

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

def ranked_components_plot_ax(labels, values, title, ax, color, abs_vals=True):
    ''' pca components plot, ranked by highest loading

    inputs: labels: (list or pandas.core.indexes.base.Index)
            values: (numpy array)
            title: (string)
            ax: (matplotlib axis)
            abs_vals: flag, use absolute value or not'''

    # sort components by value
    if abs_vals:
        zipped = zip(labels, abs(values))
    else:
        zipped = zip(labels, values)
    srted = sorted(zipped, key=lambda x: x[1], reverse=True)

    # labels, values
    arr = np.array(srted)
    labels = arr[:,0]
    vals = arr[:,1]
    vals = vals.astype(float)
    # indices
    idx = np.arange(len(vals))
    idx = np.flip(idx, axis=0)

    # plot
    ax.barh(idx, vals, color=color)
    ax.set_yticks(idx)
    ax.set_yticklabels(labels)
    ax.set_title(title)

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

    fig, ax = plt.subplots(1,2)
    idx_axes = [0,1]
    colors = ['b','g']
    for pc, idx_ax, c in zip(pcs, idx_axes, colors):
        ranked_components_plot_ax(species_df.columns, pcs[pc], pc, ax[idx_ax], c, abs_vals=True)
    plt.tight_layout()
    plt.show()
