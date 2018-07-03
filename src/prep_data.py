import pandas as pd
import numpy as np
import pickle

if __name__=='__main__':
    # read from file
    df = pd.read_csv('../data/Cal_climate_exp_phenotyped.csv')

    # df of all data, with species as index
    df_all = df.copy()
    df_all.set_index('species', inplace=True)
    df_all.drop(['SP', 'Phenotype'], axis=1, inplace=True)

    # df aggregated on species
    cols = list(df.columns.drop(['Phenotype', 'SP', 'species']))
    species_df = df.groupby('species')[cols].mean()

    # save to pickle
    pickle.dump(df_all, open('pkl/df_all.pkl', 'wb'))
    pickle.dump(species_df, open('pkl/species_df.pkl', 'wb'))
