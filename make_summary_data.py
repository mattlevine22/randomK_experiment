import os, sys
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import random
from pdb import set_trace as bp

N_Ks = 50

# Set locations
master_file = './raw_data/master_file.pkl' # this is primary raw data from experiments
voxel_dir = './raw_data/voxel_averages' # this is jacob's library of targets from voxel averaging
output_data_dir = './summary_data' # this is output directory for summary data that we build in this script.

def make_Kij_names(m):
    """
    Create Kij names for ordering parameters
    """

    # n_accesory = m - n_input
    return [f'K{i[0]}{i[1]}' for i in itertools.combinations_with_replacement(range(1, m+1), 2)]

## Write label keys for K
K_names = {m: make_Kij_names(m) for m in range(3, 13)}
a_names = {m: [f'x{i}' for i in range(2, m+1)] for m in range(3, 13)}
# Write name dicts to file
with open(os.path.join(output_data_dir, 'name_dict.pkl'), 'wb') as f:
    pickle.dump({'K_names': K_names, 'a_names': a_names}, f)

## aggregate target curves
df_targets = pd.DataFrame()
for m in range(3, 13):
    foo_file = os.path.join(voxel_dir, '{}M_voxel_averages.npy'.format(m))
    x = np.load(foo_file)
    new_df = pd.DataFrame(x)
    new_df['m'] = m
    new_df['targetID'] = new_df.index

    if len(df_targets) == 0:
        df_targets = new_df
    else:
        df_targets = pd.concat([df_targets, new_df])
df_targets = df_targets.reset_index(drop=True)
N_targets = df_targets.groupby('m').targetID.max() + 1

## aggregate K values
with open(master_file, 'rb') as f:
    x = pickle.load(f)

# K_dict[m] : N_Ks x N_dimers
K_dict = {m: np.nan*np.ones((N_Ks, int(m*(m+1)/2))) for m in range(3, 13)}

# a_dict[m] : N_Ks x N_dimers x N_targets x N_accesory
a_dict = {m: np.nan*np.ones((N_Ks, int(m*(m+1)/2), N_targets[m], m-1)) for m in range(3, 13)}

keep_keys = ['m', 'targetID', 'KID', 'dimerID']
df = pd.DataFrame()
# iteritems = random.sample(x.items(), 10000)
iteritems = x.items()
for key, value in tqdm(iteritems):
    atts = {g.split('-')[0]: float(g.split('-')[1]) for g in key.split('_')}

    # update summary dict
    foo = {k: atts[k] for k in keep_keys}
    foo['Linf'] = float(value['Linf'])
    foo['MSE'] = float(value['MSE'])
    df = pd.concat( [df, pd.DataFrame.from_dict([foo]) ])

    # update K_dict
    m = int(atts['m'])
    KID = int(atts['KID'])
    targetID = int(atts['targetID'])
    dimerID = int(atts['dimerID'])

    # K_dict[m] : N_Ks x N_dimers
    K_dict[m][KID] = np.squeeze(value['K'])

    # update a_dict
    # a_dict[m] : N_Ks x N_dimers x N_targets x N_accesory
    a_dict[m][KID, dimerID, targetID] = np.squeeze(value['a'])

    # update df_targets to show that this target has been used
    df_targets.loc[(df_targets.m==m) & (df_targets.targetID==targetID), 'used'] = True

df['goodenough'] = df.Linf <= 1

# convert df columns m, targetID, KID, dimerID to int
int_cols = ['m', 'targetID', 'KID', 'dimerID']
df[int_cols] = df[int_cols].astype(int)

# move columns ["m", "targetID", "used"] to the front
cols = list(df_targets.columns)
cols = cols[-3:] + cols[:-3]
df_targets = df_targets[cols]
# write to csv  
df_targets.to_csv(os.path.join(output_data_dir,'targets.csv'), index=False)

# write df to csv
df.to_csv(os.path.join(output_data_dir,'summary.csv'), index=False)

# write K and a to file
with open(os.path.join(output_data_dir,'K_random.pkl'), 'wb') as f:
    pickle.dump(K_dict, f)

with open(os.path.join(output_data_dir,'a_opt.pkl'), 'wb') as f:
    pickle.dump(a_dict, f)
