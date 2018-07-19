import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance
import pdb

# imports for standard font size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
import matplotlib
matplotlib.rc('font', **font)

def scaled_df(df):
    ''' returns a pandas dataframe with data scaled by StandardScaler
    requires: sklearn StandardScaler()
    input: pandas df
    output: pandas df
    '''
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

def kmeans_labeled(df, n_clusters):
    ''' fit kmeans clustering to dataframe, return dataframe and
    fitted estimator'''
    kmm = KMeans(n_clusters = n_clusters)
    kmm.fit(df)
    labeled_df = df.copy()
    labeled_df['cluster'] = kmm.labels_
    return labeled_df, kmm

    def intra_cluster_distance_manual(X, labels, label_dict):
        ''' calculates euclidean distance norm for manually labeled clusters
            in two dimensions (e.g. pc1, pc2)

        inputs:
        X: np array, shape (n_obs, 2)
        labels: np array, shape (n_obs,), vals = integer labels
        label_dict: dictionary mapping integer to string labels

        output:
        print: label, norm
        '''
        label_vals = np.unique(labels) # unique labeled values
        classes = []
        for val in label_vals:
            # subset of data with same label
            X_sub = X[labels == val]
            # find center
            C = np.array([X_sub[:,0].mean(), X_sub[:,1].mean()])
            # compute euclidean distance norm
            D = distance.cdist(X_sub, C.reshape(-1,1).T, 'euclidean')
            norm = np.sqrt(np.sum(D**2))
            classes.append((val, norm))

        # apply correct labels
        for item in classes:
            label = label_dict[item[0]]
            print('{}: euclidean distance norm = {:0.3f}'.format(label, item[1]))

if __name__=='__main__':
    # read df from pickle
    species_df = pickle.load(open('pkl/species_df.pkl', 'rb'))
    phenotype = species_df.pop('Phenotype')

    # scale data
    X_scaled = scaled_df(species_df)

    # pca
    n_components = 2
    pca = PCA(n_components)
    pca.fit(X_scaled)
    X_reduced = pca.transform(X_scaled)
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

    # principal component scatter plot
    plt.scatter(X_reduced[:,0], X_reduced[:,1], marker='o', c=phenotype)
    #plt.legend(['type1', 'type2', 'type3'])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['C3', 'constitutive', 'facultative'])
    plt.tight_layout()
    plt.show()

    # intra-cluster distance for manually labeled clusters
    num_to_label = {1: 'C3', 2: 'constitutive', 3: 'facultative'}
    labels = phenotype.values
    print('\n')
    intra_cluster_distance_manual(X_reduced, labels, num_to_label)


    # clustering on pca dimension-reduced data
    kmm = KMeans(n_clusters = 3)
    kmm.fit(X_reduced)
    labels = kmm.labels_
    centers = kmm.cluster_centers_

    # intra-cluster variance
    X_labeled = np.concatenate((X_reduced, labels.reshape(-1,1)), axis=1) # concat labels
    X_sorted = X_labeled[X_labeled[:,2].argsort()] # sort by labels

    arrays = []
    for i in range(max(labels) + 1):
        X = X_labeled[X_labeled[:,2] == i]
        X = X[:,0:2] # remove labels
        C = centers[i,:] #pc1, pc2 coordinates of cluster centers
        # np euclidean norm
        dist = np.linalg.norm(X-C, ord=2) # euclidean norm
        # scipy euclidean norm
        D = distance.cdist(X, C.reshape(-1,1).T, 'euclidean')
        norm = np.sqrt(np.sum(D**2))
        #dist = distance.euclidean(X[:,], C)
        arrays.append((X, C, norm))
        print('cluster {}'.format(i))
        print('np.linalg.norm: {:0.3f}'.format(dist))
        print('euclidean norm: {:0.3f}\n'.format(norm))

    # plot pca-reduced clusters
    colors = ['b','g','r']
    i = 0
    for X, C, norm in arrays:
        plt.scatter(X[:,0], X[:,1], c=colors[i], s=10)
        plt.scatter(C[0], C[1], c=colors[i], s=40, marker='+')
        i += 1
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(['type1', 'type2', 'type3'])
    plt.show()
