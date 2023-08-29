
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

def plot_umap_labeled(data, x, y, cluster, palette, xlim=None, ylim=None, axes=False, figsize=(6,6), 
                      force_points=0.002, force_text=0.3, adjust=True, **kwargs):
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
    centroids = data.groupby('cluster')[[x,y]].median()
    texts = [plt.text(data[x], data[y], name.replace(' ', '\n'), 
                      ha='center', va='center', ma='center', size=10) 
             for name, data in centroids.iterrows()]
    if adjust:
        adjust_text(texts, data[x].values, data[y].values, 
                arrowprops=dict(arrowstyle="-", color='k', lw=0.7),
                force_points=force_points, force_text=force_text, expand_points=(1,1))