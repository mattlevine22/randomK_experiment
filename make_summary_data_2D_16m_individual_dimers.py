import os
import numpy as np
import pandas as pd
import pickle
import itertools
from tqdm import tqdm
from pdb import set_trace as bp

n_inputs = 2
N_Ks = 10
m_list = [16]

target_names = ['AND', 'NAND', 'OR', 'XOR', 'NOR', 'XNOR', 'ANOTB', 'AeqB', 'AneqB',
       'DoubleBandpass', 'ratio']

N_targets = {m: len(target_names) for m in m_list}

# Set locations
master_file = './raw_data/master_file_2D_16m_individual_dimers.pkl' # this is primary raw data from experiments
output_data_dir = './summary_data_2D_16m_individual_dimers' # this is output directory for summary data that we build in this script.

def make_Kij_names(m):
    """
    Create Kij names for ordering parameters
    """

    # n_accesory = m - n_input
    return [f'K{i[0]}{i[1]}' for i in itertools.combinations_with_replacement(range(1, m+1), 2)]

## Write label keys for K
K_names = {m: make_Kij_names(m) for m in m_list}
a_names = {m: [f'x{i}' for i in range(2, m+1)] for m in m_list}
# Write name dicts to file
with open(os.path.join(output_data_dir, 'name_dict.pkl'), 'wb') as f:
    pickle.dump({'K_names': K_names, 'a_names': a_names}, f)

## aggregate K values
with open(master_file, 'rb') as f:
    x = pickle.load(f)

# K_dict[m] : N_Ks x N_dimers
K_dict = {m: np.nan*np.ones((N_Ks, int(m*(m+1)/2))) for m in m_list}

# a_dict[m] : N_Ks x N_dimers x N_targets x N_accesory
a_dict = {m: np.nan*np.ones((N_Ks, int(m*(m+1)/2), N_targets[m], m-n_inputs)) for m in m_list}

keep_keys = ['m', 'targetID', 'KID', 'dimerID']
df = pd.DataFrame()
# iteritems = random.sample(x.items(), 10000)
iteritems = x.items()
for key, value in tqdm(iteritems):
    atts = {}
    for g in key.split('_'):
        fooval = g.split('-')[1]
        try:
            fooval = float(fooval)
        except:
            pass
        atts[g.split('-')[0]] = fooval

    # update summary dict
    foo = {k: atts[k] for k in keep_keys}
    foo['Linf'] = float(value['Linf'])
    foo['MSE'] = float(value['MSE'])
    df = pd.concat( [df, pd.DataFrame.from_dict([foo]) ])

    # update K_dict
    m = int(atts['m'])
    KID = int(atts['KID'])
    targetID = target_names.index(atts['targetID'])
    dimerID = int(atts['dimerID'])

    # K_dict[m] : N_Ks x N_dimers
    K_dict[m][KID] = np.squeeze(value['K'])

    # update a_dict
    # a_dict[m] : N_Ks x N_dimers x N_targets x N_accesory
    a_dict[m][KID, dimerID, targetID] = np.squeeze(value['a'])

df['goodenough'] = df.Linf <= 1

# convert df columns m, targetID, KID, dimerID to int
int_cols = ['m', 'KID', 'dimerID']
df[int_cols] = df[int_cols].astype(int)

# make new column with targetID as int
df['targetID_name'] = df.targetID
df['targetID'] = df.targetID.apply(lambda x: target_names.index(x))

# write df to csv
df.to_csv(os.path.join(output_data_dir,'summary.csv'), index=False)

# write K and a to file
with open(os.path.join(output_data_dir,'K_random.pkl'), 'wb') as f:
    pickle.dump(K_dict, f)

# write target_names to file
with open(os.path.join(output_data_dir,'target_names.pkl'), 'wb') as f:
    pickle.dump(target_names, f)

with open(os.path.join(output_data_dir,'a_opt.pkl'), 'wb') as f:
    pickle.dump(a_dict, f)
