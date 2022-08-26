from posixpath import join
import pandas as pd
import os.path

def load_shiny_lastmap(species):
    directory = shiny_directory(species)
    path = os.path.join(directory, 'mapping.df.lastmap.csv')
    shiny_df = pd.read_csv(path)
    return shiny_df
    
def load_shiny_data(species, directory=None, csv_path=None, feather_path=None, rda_path=None, drop_offpipeline=True, nms_pass=True):
    shiny_df = _load_shiny_data(species, directory, csv_path, feather_path, rda_path)
    shiny_df = filter_shiny_data(shiny_df, drop_offpipeline=drop_offpipeline, nms_pass=nms_pass)
    return shiny_df

def _load_shiny_data(species=None, directory=None, csv_path=None, feather_path=None, rda_path=None):
    if csv_path:
        shiny_df = pd.read_csv(csv_path)
    elif feather_path:
        shiny_df = pd.read_feather(feather_path)
    elif rda_path:
        import pyreadr
        shiny_df = pyreadr.read_r(rda_path)[None]
    else:
        directory = directory or shiny_directory(species)
        path = os.path.join(directory, 'anno.feather')
        shiny_df = pd.read_feather(path)
    shiny_df.drop(columns=[col for col in shiny_df.columns 
        if col.endswith('_color') or (col.endswith('_id') and not col in ['spec_id','sample_id'])], 
        inplace=True)
    shiny_df.rename(axis=1, mapper=lambda col: col.replace('_label',''), inplace=True)
    shiny_df.replace('ZZ_Missing', float('nan'), inplace=True)
    if 'spec_id' in shiny_df.columns:
        shiny_df = shiny_df.dropna(subset=['spec_id'])
        assert shiny_df['spec_id'].is_unique
        shiny_df.index = shiny_df['spec_id'].astype(int)

    return shiny_df

def shiny_directory(species):
    feather_path = '/allen/programs/celltypes/workgroups/rnaseqanalysis/shiny/patch_seq/star/{}'
    # Maybe hard-code a date since columns may change with version?
    if species=='human':
        current = 'human/human_patchseq_MTG_20220428'
        # current = 'human/human_patchseq_MTG_current'
        feather_path = feather_path.format(current)
    elif species=='mouse':
        current = 'mouse_patchseq_VISp_20220428_collapsed40_cpm'
        # current = 'mouse_patchseq_VISp_current'
        feather_path = feather_path.format(current)
    return feather_path

def filter_shiny_data(shiny_df, drop_offpipeline=True, nms_pass=True):
    project_col = 'cell_specimen_project'
    if drop_offpipeline:
        # ends with MET, not METx, METc etc
        shiny_df = shiny_df[~shiny_df[project_col].isna() & shiny_df[project_col].str.endswith("MET")]
    if nms_pass:
        shiny_df = shiny_df[shiny_df['Norm_Marker_Sum.0.4']=='TRUE']
    return shiny_df

def load_genes_shiny(genes, species=None, directory=None, csv_path=None, drop_offpipeline=False, nms_pass=False, join_on='sample_id'):
    shiny_df = _load_shiny_data(species, directory, csv_path)

    
    genes_df = pd.read_feather(os.path.join(directory, 'data.feather'), columns=genes+[join_on]).set_index(join_on)
    shiny_df = shiny_df.join(genes_df, on=join_on)
    shiny_df = filter_shiny_data(shiny_df, drop_offpipeline=drop_offpipeline, nms_pass=nms_pass)
    return shiny_df

def get_shiny_palette(species, add_other=True):
    directory = shiny_directory(species)
    path = os.path.join(directory, 'anno.feather')
    df = pd.read_feather(path)
    key = "cluster"
    colors = df.groupby(f'{key}_label')[f'{key}_color'].first()
    palette = colors.to_dict()
    if add_other:
        palette['other'] = 'grey'
    return palette