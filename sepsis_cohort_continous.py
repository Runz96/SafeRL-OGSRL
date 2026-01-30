"""
This file is sourced and modified from: https://github.com/microsoft/mimic_sepsis
"""

import numpy as np
import pandas as pd
import pyprind
import argparse
from scipy import stats

process_raw = True

MIMICtable = pd.read_csv('MIMICtable.csv')  # Load the MIMIC data

#################   Define columns and normalize data    ######################
# Meta-data columns
colmeta = ['presumed_onset', 'charttime', 'icustayid']
# Binary features (note: max_dose_vaso is kept as continuous)
colbin = ['gender', 'mechvent', 'max_dose_vaso', 're_admission']
# Features to z-normalize
colnorm = ['age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1',
           'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb', 
           'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2',
           'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index',
           'PaO2_FiO2', 'cumulated_balance']
# Features to log-normalize
collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR',
          'input_total', 'input_4hourly', 'output_total', 'output_4hourly']

# Create RAW and normalized datasets
MIMICraw = MIMICtable[colmeta + colbin + colnorm + collog]

MIMICzs = np.hstack([
    MIMICtable[colmeta].values, 
    MIMICtable[colbin].values - 0.5, 
    stats.zscore(MIMICtable[colnorm].values), 
    stats.zscore(np.log(0.1 + MIMICtable[collog].values))
])

# Additional transformations (as in your original code)
MIMICzs[:, 5] = np.log(MIMICzs[:, 5] + 0.6)   # Transform max_dose_vaso if desired
MIMICzs[:, 46] = 2 * MIMICzs[:, 46]            # Increase weight of this variable

#---------------------- Process Normalized Data ----------------------#
# Create DataFrames for meta and observation features
MIMICzs = pd.DataFrame(MIMICzs, columns=colmeta + colbin + colnorm + collog)
meta_df = pd.DataFrame(MIMICzs[colmeta].values, columns=colmeta)
ob_df = pd.DataFrame(MIMICzs[colbin + colnorm + collog].values, columns=colbin + colnorm + collog)

# Use the "bloc" column (assumed to be present) to determine trajectory boundaries
raw_data_df = MIMICtable.copy()
raw_data_df['traj'] = (raw_data_df['bloc'] == 1).cumsum().values
meta_df['traj'] = raw_data_df['traj']
ob_df['traj'] = raw_data_df['traj']

# Define outcome key (e.g., for mortality)
outcome_key = 'died_within_48h_of_out_time'  # or 'mortality_90d'
meta_cols = meta_df.columns.tolist()
# Remove "traj" from observation columns to avoid duplication in the final output.
ob_cols = [col for col in ob_df.columns.tolist() if col != 'traj']

trajectories = raw_data_df['traj'].unique()
data = {}
data['meta_cols'] = meta_cols
data['obs_cols'] = ob_cols
data['traj'] = {}

print('Sepsis Cohort -- Making trajectory data (normalized)')
bar = pyprind.ProgBar(len(trajectories))
for traj in trajectories:
    bar.update()
    data['traj'][traj] = {}
    meta_array = meta_df[meta_df['traj'] == traj][meta_cols].values.T
    obs_array = ob_df[ob_df['traj'] == traj][ob_cols].values.T
    data['traj'][traj]['meta'] = meta_array
    data['traj'][traj]['obs'] = obs_array
    # Keep outcome and reward computations:
    n_steps = meta_array.shape[1]  # number of time steps
    data['traj'][traj]['outcome'] = raw_data_df[raw_data_df['traj'] == traj][outcome_key].values[0]
    data['traj'][traj]['rewards'] = np.zeros(n_steps)
    data['traj'][traj]['rewards'][-1] = (1 - 2 * data['traj'][traj]['outcome'])

print('Sepsis Cohort -- Making final normalized output file')
col_names = ['traj', 'step']
col_names.extend(['m:' + i for i in data['meta_cols']])
col_names.extend(['o:' + i for i in data['obs_cols']])
col_names.append('r:reward')
all_data = []
bar = pyprind.ProgBar(len(data['traj'].keys()))
for traj in data['traj'].keys():
    bar.update()
    n_steps = data['traj'][traj]['meta'].shape[1]
    for ctr in range(n_steps):
        row = [traj, ctr]
        for m in range(data['traj'][traj]['meta'].shape[0]):
            row.append(data['traj'][traj]['meta'][m, ctr])
        for o in range(data['traj'][traj]['obs'].shape[0]):
            row.append(data['traj'][traj]['obs'][o, ctr])
        row.append(data['traj'][traj]['rewards'][ctr])
        all_data.append(row)

df = pd.DataFrame(all_data, columns=col_names)
df.to_csv('sepsis_final_data_withTimes_continuous.csv', index=False)

#---------------------- Process RAW Data (if desired) ----------------------#
if process_raw:
    raw_df = pd.DataFrame(MIMICraw, columns=colmeta + colbin + colnorm + collog)
    meta_df = pd.DataFrame(raw_df[colmeta].values, columns=colmeta)
    ob_df = pd.DataFrame(raw_df[colbin + colnorm + collog].values, columns=colbin + colnorm + collog)
    raw_data_df = MIMICtable.copy()
    raw_data_df['traj'] = (raw_data_df['bloc'] == 1).cumsum().values
    meta_df['traj'] = raw_data_df['traj']
    ob_df['traj'] = raw_data_df['traj']
    
    trajectories = raw_data_df['traj'].unique()
    data = {}
    data['meta_cols'] = meta_df.columns.tolist()
    data['obs_cols'] = [col for col in ob_df.columns.tolist() if col != 'traj']
    data['traj'] = {}
    
    print('Sepsis Cohort -- Making RAW trajectory data')
    bar = pyprind.ProgBar(len(trajectories))
    for traj in trajectories:
        bar.update()
        data['traj'][traj] = {}
        meta_array = meta_df[meta_df['traj'] == traj][data['meta_cols']].values.T
        obs_array = ob_df[ob_df['traj'] == traj][data['obs_cols']].values.T
        data['traj'][traj]['meta'] = meta_array
        data['traj'][traj]['obs'] = obs_array
        n_steps = meta_array.shape[1]
        data['traj'][traj]['outcome'] = raw_data_df[raw_data_df['traj'] == traj][outcome_key].values[0]
        data['traj'][traj]['rewards'] = np.zeros(n_steps)
        data['traj'][traj]['rewards'][-1] = (1 - 2 * data['traj'][traj]['outcome'])
    
    print('Sepsis Cohort -- Making final RAW output file')
    col_names = ['traj', 'step']
    col_names.extend(['m:' + i for i in data['meta_cols']])
    col_names.extend(['o:' + i for i in data['obs_cols']])
    col_names.append('r:reward')
    all_data = []
    bar = pyprind.ProgBar(len(data['traj'].keys()))
    for traj in data['traj'].keys():
        bar.update()
        n_steps = data['traj'][traj]['meta'].shape[1]
        for ctr in range(n_steps):
            row = [traj, ctr]
            for m in range(data['traj'][traj]['meta'].shape[0]):
                row.append(data['traj'][traj]['meta'][m, ctr])
            for o in range(data['traj'][traj]['obs'].shape[0]):
                row.append(data['traj'][traj]['obs'][o, ctr])
            row.append(data['traj'][traj]['rewards'][ctr])
            all_data.append(row)
    
    df = pd.DataFrame(all_data, columns=col_names)
    df.to_csv('sepsis_final_data_RAW_withTimes_continuous.csv', index=False)

