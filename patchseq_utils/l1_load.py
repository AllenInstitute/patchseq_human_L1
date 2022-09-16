import pandas as pd
from pathlib import Path
from pandas.api.types import CategoricalDtype
from . import shiny, lims
from .util import *


datadir = Path("/home/tom.chartrand/projects/data/u01/")
projectdir = Path('/home/tom.chartrand/projects/human_l1')
figdir = projectdir/'figures'


cluster = 't-type'
palette_human = shiny.get_shiny_palette('human')
palette_human = {names_update[shorten_name(key)]: val for key, val in palette_human.items()
                if shorten_name(key) in names_update}
palette_mouse = shiny.get_shiny_palette('mouse')
palette_subclass =  {
'LAMP5': '#e68553',
'MC4R': '#ae1e3d',
'PAX6': '#e3b5c7',
'L1 VIP': '#6c00bf',
'other': 'grey',
}
species_palette = palette = {
    'human':'tab:pink',
    'mouse':'tab:grey',
}
date = "2022_06_22"
date = "2022_09_06"
human_df = pd.read_csv(f"~/projects/human_l1/human_l1_dataset_{date}.csv", index_col=0,
                      dtype = {'layer_lims': str, 'target_layer': str})
human_ephys = pd.read_csv(datadir/"aibs"/"features_E.csv", index_col=0)
tamas_ephys = pd.read_csv(datadir/"tamas"/"features_E.csv", index_col=0)
mansvelder_ephys = pd.read_csv(datadir/"mansvelder"/"features_E.csv", index_col=0)
human_depth = pd.read_csv(projectdir/'human_layer_depths_2022_09_02.csv', index_col='specimen_id').drop(columns=['Unnamed: 0'])
human_morph = pd.read_csv(projectdir/'RawFeatureWide_human+derivatives.csv', index_col='specimen_id').drop(columns=['Unnamed: 0'])

mouse_df = pd.read_csv(projectdir/f"mouse_l1_dataset_{date}.csv", index_col=0)
mouse_ephys = pd.read_csv(datadir/"aibs_mouse/features_E.csv", index_col=0)
mouse_depth = pd.read_csv(projectdir/'mouse_layer_depths_2022_09_02.csv', index_col='specimen_id').drop(columns=['Unnamed: 0'])
ccf_depth = pd.read_csv(projectdir/'mouse_l1_dataset_2021_10_25_ccf_aligned_depths.csv', index_col=0)
mouse_morph = pd.read_csv(projectdir/'RawFeatureWide_mouse+derivatives.csv', index_col='specimen_id').drop(columns=['Unnamed: 0'])

ephys_features = [feat for feat in human_ephys.columns
                  # .intersection(mouse_ephys.columns) 
                  if 'qc' not in feat and 'fail' not in feat]
morph_features = human_morph.columns

mouse_depth.layer = mouse_depth.layer.fillna('').apply(lambda x: x[5:] if x else None)
human_depth.layer = human_depth.layer.fillna('').apply(lambda x: x[5:] if x else None)

homology = {
    'LAMP5':{
        'mouse':['Lamp5 Lsp1', 'Lamp5 Plch2 Dock5', 'Lamp5 Ntn1 Npy2r'],
        'human':['Inh L1-6 LAMP5 LCP2', 'Inh L1-2 LAMP5 DBP', 'Inh L1 SST NMBR']
    },
    'MC4R':{
        'mouse':['Lamp5 Fam19a1 Tmem182', 'Lamp5 Fam19a1 Pax6'],
        'human':['Inh L1 SST CHRNA4', 'Inh L1-2 GAD1 MC4R']
    },
    'PAX6':{
        'mouse':['Lamp5 Krt73'],
        'human':['Inh L1-2 PAX6 CDH12', 'Inh L1-2 PAX6 TNFAIP8L3']
    },
    'L1 VIP':{
        'mouse':['Sncg Vip Nptx2', 'Vip Col15a1 Pde1a'],
        'human':[ 'Inh L1-2 VIP TSPAN12']
    },
    'other':{
        'mouse':[],
        'human':['Inh L1-2 SST BAGE2', 'Inh L1-2 VIP PCDH20']
    },
}
#     'Vip/Sncg':{
#         'mouse':['Sncg Gpr50', 'Sncg Slc17a8', 'Sncg Vip Itih5', 'Sncg Vip Nptx2', 'Vip Col15a1 Pde1a'],
#         'human':['Inh L1-3 PAX6 SYT6', 'Inh L1-2 VIP TSPAN12']
#     },
homology_types = CategoricalDtype(categories=homology.keys(), ordered=True)

homology_mapping_human = {names_update[shorten_name(ttype)]: hom_type for hom_type in homology for ttype in homology[hom_type]['human']}
homology_mapping_mouse = {ttype: hom_type for hom_type in homology for ttype in homology[hom_type]['mouse']}
ttypes_human = CategoricalDtype(categories=homology_mapping_human.keys(), ordered=True)
ttypes_mouse = CategoricalDtype(categories=homology_mapping_mouse.keys(), ordered=True)
l1_types_human = ttypes_human.categories
l1_types_mouse = ttypes_mouse.categories

human_df[cluster] = human_df[cluster].astype(ttypes_human)
human_df['homology_type'] = human_df[cluster].map(homology_mapping_human).astype(homology_types)
mouse_df[cluster] = mouse_df[cluster].astype(ttypes_mouse)
mouse_df['homology_type'] = mouse_df[cluster].map(homology_mapping_mouse).astype(homology_types)

# human_df[cluster] = human_df[cluster].map(names_update).astype(ttypes_human)
# human_df['homology_type'] = human_df[cluster].map(homology_mapping_human).astype(homology_types)
# mouse_df[cluster] = mouse_df[cluster].astype(ttypes_mouse)
# mouse_df['homology_type'] = mouse_df[cluster].map(homology_mapping_mouse).astype(homology_types)

human_df = pd.concat([
    human_df[human_df.collaborator=='Gabor'].join(tamas_ephys, on='sample_id'),
    human_df[human_df.collaborator=='Huib'].join(mansvelder_ephys, on='sample_id'),
    human_df[human_df.collaborator.isna()].join(human_ephys).assign(collaborator='AIBS')
], sort=False)

# only column overlap is layer, which is from LIMS?
human_df = human_df.join(human_depth, lsuffix='_lims').join(human_morph)
mouse_df = (mouse_df.join(mouse_ephys)
            .join(mouse_depth, lsuffix='_lims')
            .join(ccf_depth)
            .join(mouse_morph))

# TODO: fix this elsewhere
human_df.loc[lambda df: df['first_isi_inv_hero']>1000, 'first_isi_inv_hero'] = np.nan
human_df.loc[lambda df: df['first_isi_inv_rheo']>1000, 'first_isi_inv_rheo'] = np.nan
human_df.loc[lambda df: df['input_resistance_ss']>1000, 'input_resistance_ss'] = np.nan
human_df.loc[lambda df: df['input_resistance']>1000, 'input_resistance'] = np.nan

# remove Tx-only cells
human_df = human_df.query("has_morph | layer==layer | failed_fx_long_squares==False")
