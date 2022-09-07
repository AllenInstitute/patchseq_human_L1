from . import analysis as utils
from .plot import sweeps as ps
from .plot import morphology as pm
from .l1_load import palette_human, palette_subclass
from .util import shorten_name, names_update, remove_unused_categories
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

def plot_umap_labeled(data, x, y, cluster, palette, xlim=None, ylim=None, axes=False, figsize=(6,6), **kwargs):
    plt.figure(figsize=figsize)
    sns.scatterplot(data=data.sample(frac=1), x=x, y=y, hue=cluster, palette=palette, legend=None, **kwargs)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if not axes:
        sns.despine(left=True, bottom=True)
        plt.xticks([])
        plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    centroids = data.groupby('cluster')[[x,y]].mean()
    texts = [plt.text(data[x], data[y], name.replace(' ', '\n'), 
                      ha='center', va='center', ma='center', size=10) 
             for name, data in centroids.iterrows()]
    adjust_text(texts, data[x].values, data[y].values, 
                arrowprops=dict(arrowstyle="-", color='k', lw=0.7),
                force_points=0.002, force_text=0.3, expand_points=(1,1))

def box_strip(data, x, y, hue=None, size=3, ax=None, strip_width=0.2, label_counts=False, legend=False, notch=False,
               figsize=(4,2.5), leg_kws={}, transparent=True, **kwargs):
    if not hasattr(data[x], 'cat'):
        # make column ordered categorical
        data[x] = data[x].astype('category')
        
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if ax=='gca':
        ax = plt.gca()
    args = dict(showfliers=False)
    args.update(kwargs)
    if notch:
        args.update(notch=True, bootstrap=1000)
    sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, **args)
    args = dict(s=size, alpha=0.6)
    args.update(kwargs)
    sns.stripplot(data=data, x=x, y=y, hue=hue, ax=ax, jitter=strip_width, **args)
    # sns.scatterplot(data=data, x=x, y=y, hue=None, ax=ax, x_jitter=strip_width, **args)
    handles, labels = ax.get_legend_handles_labels()
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

    if transparent:
        utils.outline_boxplot(ax)
    
def plot_nested_comparisons(data, x, y, compare, test_subgroups=True, fdr_method=None,
                           label=None, xlabel=False, bar=True, sig_label='stars', ax=None, **kwargs):
    dodge_delta = 0.2
    box_strip(data, x, y, hue=compare, dodge=True, ax=ax, **kwargs)
    ax = ax or plt.gca()
    if not xlabel:
        ax.set_xlabel(None)
    if label:
        ax.set_title(label)
        ax.set_ylabel(None)
    if test_subgroups:
        y0 = ax.get_ylim()[1]
        res = utils.subgroup_comparisons(data, [y], x, compare, fdr_method=fdr_method)
        sig_diffs = res.sig_groups.values[0].split(', ')
        for group in sig_diffs:
            if len(group)>0:
                i = data[x].cat.categories.get_loc(group)
                x_pair = [i-dodge_delta, i+dodge_delta]
                utils.plot_sig_bar(res[group].values[0], y0, x_pair, bar=bar, label=sig_label, ax=ax)

def plot_subclass_focus(df, y, x, ax, subclass, label=None, cluster="t-type",
                        palette=palette_subclass, palette_fine=palette_human, **kwargs):
    s = 4
    data=df.query(f"{x}=='{subclass}'").copy().pipe(remove_unused_categories)
    box_strip(data=data, ax=ax, x=x, size=s,
                   y=y, hue=cluster, palette=palette_fine, dodge=True, legend=False, transparent=False,
                  order=palette.keys())

    data=df.query(f"{x}!='{subclass}'").copy()
    utils.plot_box_cluster_feature(data, y, x, x_fine=cluster, ax=ax, size=s, label=label,
                                   drop_box='other', label_counts=False,
                                   palette_fine=palette_fine, palette=palette, **kwargs)

def plot_scatter(*args, legend=False, figsize=(8,8), **kwargs):
    plt.figure(figsize=figsize)
    sns.scatterplot(*args, legend=legend, 
                     **kwargs)
    sns.despine()
    plt.xticks([])
    plt.yticks([])
    
#     trace/morph panels
file_mappings = {
    'Gabor': lambda sample_id: f"/home/tom.chartrand/projects/data/u01/tamas/converted/{sample_id}.nwb",
    'Huib': lambda sample_id: f"/home/tom.chartrand/projects/data/u01/mansvelder/final/{sample_id}.nwb",
    'AIBS':None
}
output_mappings = {
    'Gabor': lambda sample_id: f"/home/tom.chartrand/projects/data/u01/tamas/output/{sample_id}.json",
    'Huib': lambda sample_id: f"/home/tom.chartrand/projects/data/u01/mansvelder/output/{sample_id}.json",
    'AIBS': lambda sample_id: f"/home/tom.chartrand/projects/data/u01/aibs/output/{sample_id}.json"
}

def plot_trace_morph(cell, df, palette=palette_human, ttype=None, save=False, plot_peri=False, scale_factor=100, scalebar=True, **kwargs):
    cell_entry = df.loc[cell]
    ttype_name = cell_entry["t-type"]
    ttype = ttype or ttype_name.split(' ')[-1]
    if cell_entry['has_morph']:
        print(f"{cell} morph")
        try:
            pm.plot_cell_lims(cell, scale_factor=scale_factor, scalebar=scalebar, 
                              color=palette[ttype_name], **kwargs)
            plt.show()
        except Exception as exc:
            print(exc)

    
    if cell_entry['has_ephys']:
        print(f"{cell} ephys")
        try:
            if 'collaborator' not in cell_entry or cell_entry['collaborator']=='AIBS':
                dataset, sweeps = ps.get_dataset_sweeps(cell, lims_sweep_info=True, qc_sweeps=True)
            else:
                dataset, sweeps = ps.get_dataset_sweeps(cell, lims_sweep_info=False, qc_sweeps=True, 
                                                        path=file_mappings[cell_entry['collaborator']](
                                                        cell_entry['sample_id']))
    #         ps.plot_hero(dataset, sweeps, color=palette_human[ttype_name])
    #         ps.plot_sag(dataset, sweeps, color=palette_human[ttype_name], n_max=1)
            if len(sweeps) > 0:
                ps.plot_sweep_panel(dataset, sweeps, 
                         plot_peri=plot_peri,color=palette[ttype_name])
                plt.show()
        except Exception as exc:
            print(exc)