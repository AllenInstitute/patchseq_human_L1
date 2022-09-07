import numpy as np
import re
import pandas as pd
from pathlib import Path
from . import lims, shiny
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
    data = remove_unused_categories(data)
    return data

def subset_features(data, features, complete_frac=0.8):
    complete_features = np.array(features)[
        (data[features].notna().mean(axis=0) > complete_frac)]
    return complete_features

def remove_unused_categories(df: pd.DataFrame):
    for c in df.columns:
        if pd.api.types.is_categorical_dtype(df[c]):
            df[c] = df[c].cat.remove_unused_categories()
    return df

def feature_name(feat):
    feat = feat.replace('norm_','')
    feat = feat.replace('basal_','')
    feat = feat.replace('calculate_','')
    feat = feat.replace('number_of_','#_')
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
    elif 'ir_late' in feat:
        name = 'irregularity'
    else:
        parts = feat.split('_')
        if parts[-1] in ['hero','rheo','ramp','short_square','mean','none']:
            parts = parts[:-1]
#         if parts[0] in ['norm']:
#             parts = parts[1:]
        name = ' '.join(parts)
    return name

def short_feat(feat):
    if 'adapt_ratio' in feat:
        return feat[0]+'AR'
    elif 'ahp_delay' in feat:
        return 'AHP dt'
    elif 'trough_deltav' in feat:
        return 'AHP dV'
    elif 'trough_slowdeltav' in feat:
        return 'vTrough'
    elif 'input_resistance' in feat:
        return 'Rin'
    elif 'fi_fit' in feat:
        return 'f/I'
    elif 'upstroke_downstroke' in feat:
        return 'u/d'
    elif 'upstroke' in feat:
        return 'up'
    elif 'depol' in feat:
        return 'hump'
    elif 'sag_' in feat:
        return 'sag'+feat[4].upper()
    elif 'width' in feat:
        return 'width'
    elif 'postap_slope' in feat:
        return 'dVpost'
    elif 'ir_late' in feat:
        return 'IR'
    elif 'isi_cv' in feat:
        return 'ISI CV'
    elif 'adapt' in feat:
        return 'AI'
    elif 'rheo' in feat or 'threshold_i' in feat:
        return 'rheo'
    else:
        return feat
    
def get_full_human_metadata_df():
    shiny_human = get_shiny('human')
    # filter out unmarked non-human cells, culture
    shiny_human = shiny_human.loc[
        lambda df: 
        df.cell_name.str.lower().str.startswith('h') &
        (df['collection'] != "FACS") &
        df['project_code'].isin(["hIVSCC-MET", "hIVSCC-METx", "H301", "H301x"]) ]

    shiny_human['has_morph'] = shiny_human.swc_path.notna()
    shiny_human['ephys_format'] = shiny_human.nwb_path.notna().apply(lambda x: 'lims_nwb' if x else None)
    shiny_human['target_layer'] = shiny_human.layer.fillna(shiny_human.roi).fillna('').apply(get_num)
    # shiny_human.drop(columns=['layer'], inplace=True)
    shiny_human.rename(columns={'layer':'old_layer'}, inplace=True)
    shiny_human['tx_qc'] =  shiny_human.apply(lambda df:
                                       ('GABAergic' in df.broad_class) &
                                        (df.contam_sum < 2) &
                                        (df.rna_amplification_pass_fail == "Pass")
                                       , axis=1)
    return shiny_human


def get_full_mouse_metadata_df():
    shiny_mouse = get_shiny('mouse')

    # this filter or offpipeline (project code MET)
    shiny_mouse = shiny_mouse.loc[
        lambda df:  
        (df['collection'] == "Patch-seq Production") &
        df['project_code'].isin(['mIVSCC-MET', 'mIVSCC-METx']) 
        & (df.structure.str.contains('VIS') | df.structure.str.contains('TEa'))
    ]
    shiny_mouse['target_layer'] = shiny_mouse.layer.fillna('').apply(get_num)
    shiny_mouse.drop(columns=['layer'], inplace=True)

    shiny_mouse['layer_ccf'] = shiny_mouse['structure'].fillna('').apply(get_num)
    shiny_mouse['cortical_area_ccf'] = shiny_mouse['structure'].fillna('').apply(not_num)
    shiny_mouse['has_morph'] = shiny_mouse.swc_path.notna()
    shiny_mouse['has_ephys'] = shiny_mouse.nwb_path.notna()

    shiny_mouse['tx_qc'] =  shiny_mouse.apply(lambda df:
                                        ('GABAergic' in df.broad_class) &
                                        (df.Tree_call!='PoorQ') &
                                        (df.rna_amp_pass_fail == "Pass")
                                       , axis=1)
    return shiny_df


def get_shiny(species, nms_pass=True):
    cluster = 't-type'
    shiny_df = shiny.load_shiny_data(species, drop_offpipeline=False, nms_pass=nms_pass)
    shiny_df = shiny_df.loc[lambda df: (df.broad_class.str.contains('GABAergic'))]

    lims_df = lims.get_cells_df(cells_list=shiny_df.index)
    lims_df.layer = lims_df.layer.apply(lambda x: x.split(' ')[-1] if x else x)
    shiny_df = (shiny_df
                .join(lims_df, rsuffix='_lims')
                .assign(species=species))
    if species=='human':
        shiny_df[cluster] = shiny_df['topLeaf'].map(shorten_name)
    else:
        shiny_df[cluster] = shiny_df['topLeaf']
    shiny_df['slice'] = shiny_df['cell_name'].map(
        lambda x: '.'.join(x.split('.')[:-1]))
    return shiny_df