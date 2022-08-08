import numpy as np
import re
import pandas as pd
from pathlib import Path
projectdir = Path('/home/tom.chartrand/projects/human_l1')
names_df = pd.read_csv(projectdir/"human_MTG_cluster_conversion.csv")

def get_num(x):
    x = re.search(r'[0-9/]+', x)
    return x if x is None else x.group()

def not_num(x):
    x = re.search(r'[^0-9/]+', x)
    return x if x is None else x.group()

def trim_parentheses(name):
    name = name[:name.find(' (')] if '(' in name else name
    return name

def shorten_name(name):
    name = trim_parentheses(name)
    if 'Exc L' in name or 'Inh L' in name:
        name = ' '.join(name.split(' ')[2:])
    return name

names_update = names_df.set_index(names_df['patchseq_cluster'].map(shorten_name))['current_FACS_cluster'].map(shorten_name).to_dict()

def subset_data(human_df, ephys_features, ephys_frac_min=0, cluster_min=5, cluster='t-type'):
    data = human_df.loc[
        human_df[ephys_features].notna().mean(axis=1) > ephys_frac_min]
    clusters = (data[cluster].value_counts(dropna=False).loc[lambda x: x>=cluster_min]).index.values
    data = data[data[cluster].isin(clusters)]
    data[cluster].cat.remove_unused_categories(inplace=True)
    clusters = data[cluster].cat.categories
    return data, clusters

def subset_features(data, features, complete_frac=0.8):
    complete_features = np.array(features)[
        (data[features].notna().mean(axis=0) > complete_frac)]
    return complete_features

def feature_name(feat):
    feat = feat.replace('norm_','')
    if 'adapt_ratio' in feat:
        if 'fast_trough' in feat:
            name = 'AHP'
        else:
            name = feat.split('_')[0] 
        name += ' adapt ratio'
    elif 'ahp_delay_ratio' in feat:
        name = 'norm trough t'
    elif 'trough_deltav' in feat:
        name = 'AHP ΔV'
    elif 'trough_slowdeltav' in feat:
        name = 'trough ΔV'
    elif 'upstroke_downstroke' in feat:
        name = 'up/down ratio'
    elif 'upstroke' in feat:
        name = 'upstroke'
    elif 'width' in feat:
        name = 'width'
    elif 'adapt' in feat:
        name = 'adapt index'
    else:
        parts = feat.split('_')
        if parts[-1] in ['hero','rheo','ramp','short_square','mean','none']:
            parts = parts[:-1]
#         if parts[0] in ['norm']:
#             parts = parts[1:]
        name = ' '.join(parts)
    return name
