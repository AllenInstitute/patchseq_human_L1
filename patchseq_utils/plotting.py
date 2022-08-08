from . import analysis as utils
from .plot import sweeps as ps
from .plot import morphology as pm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .l1_load import palette_human, human_df
from .util import shorten_name, names_update

def box_strip(data, x, y, size=3, ax=None, strip_width=0.2, label_counts=False, legend=False, notch=False,
              figsize=(4,2.5), leg_kws={}, **kwargs):
    if not hasattr(data[x], 'cat'):
        # make column ordered categorical
        data[x] = data[x].astype('category')
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if ax=='gca':
        ax = plt.gca()
    args = dict(size=size, alpha=0.6)
    args.update(kwargs)
    sns.stripplot(data=data, x=x, y=y, ax=ax, jitter=strip_width, **args)
#     sns.scatterplot(data=data, x=x, y=y, hue=x, ax=ax, x_jitter=strip_width, **args)
    handles, labels = ax.get_legend_handles_labels()
    args = dict(showfliers=False)
    args.update(kwargs)
    if notch:
        args.update(notch=True, bootstrap=1000)
    sns.boxplot(data=data, x=x, y=y, ax=ax, **args)
    if legend:
        ax.legend(handles, labels, **leg_kws)
    elif ax.get_legend():
        ax.get_legend().remove()
    sns.despine()
    
    xlabels = [text.get_text() for text in ax.get_xmajorticklabels()]
    if label_counts:
        counts = data[x].value_counts()
        xlabels = [f"{name} ({counts.loc[name]})" for name in xlabels]
    ax.set_xticklabels(xlabels, rotation=45, fontstyle='italic', ha='right')# if len(xlabels)>=8 else 'center'
    
    utils.outline_boxplot(ax)
    
def plot_scatter(*args, legend=False, figsize=(8,8), **kwargs):
    plt.figure(figsize=figsize)
    sns.scatterplot(*args, legend=legend, 
                     **kwargs)
    sns.despine()
    plt.xticks([])
    plt.yticks([])
    
#     trace/morph panels
file_mappings = {
    'Gabor': lambda sample_id: f"/home/tom.chartrand/projects/data/u01/tamas/fixed/{sample_id}.nwb",
    'Huib': lambda sample_id: f"/home/tom.chartrand/projects/data/u01/mansvelder/final/{sample_id}.nwb",
    'AIBS':None
}

def plot_trace_morph(cell, df=human_df, ttype=None, save=False, scale_factor=100, scalebar=True, **kwargs):
    cell_entry = df.loc[cell]
    ttype_name = names_update[shorten_name(cell_entry["topLeaf"])]
    ttype = ttype or ttype_name.split(' ')[-1]
    if cell_entry['has_morph']:
        print(f"{cell} morph")
        try:
            pm.plot_cell_lims(cell, scale_factor=scale_factor, scalebar=scalebar, 
                              color=palette_human[ttype_name], **kwargs)
            plt.show()
        except Exception as exc:
            print(exc)

    
    if cell_entry['has_ephys']:
        print(f"{cell} ephys")
        try:
            if cell_entry['collaborator']=='AIBS':
                dataset, sweeps = ps.get_dataset_sweeps(cell, lims_sweep_info=True, qc_sweeps=True)
            else:
                dataset, sweeps = ps.get_dataset_sweeps(cell, lims_sweep_info=False, qc_sweeps=True, 
                                                        path=file_mappings[cell_entry['collaborator']](
                                                        cell_entry['sample_id']))
    #         ps.plot_hero(dataset, sweeps, color=palette_human[ttype_name])
    #         ps.plot_sag(dataset, sweeps, color=palette_human[ttype_name], n_max=1)
            if len(sweeps) > 0:
                ps.plot_sweep_panel(dataset, sweeps, color=palette_human[ttype_name])
                plt.show()
        except Exception as exc:
            print(exc)