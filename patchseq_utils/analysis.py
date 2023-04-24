import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import combinations
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm

def ols_model(data, formula_rhs, feature, anova_type=2, cov_type='HC3'):
    metrics = ['aic', 'bic', 'fvalue', 'f_pvalue', 'llf', 'rsquared', 'rsquared_adj', 'nobs']
    formula = f"{feature} ~ {formula_rhs}"
    res = smf.ols(formula=formula, data=data).fit(cov_type=cov_type)
    fit_dict = {name: getattr(res, name) for name in metrics}

    if anova_type:
        anova = anova_lm(res, typ=anova_type)
        anova = anova.drop(index=["Residual"])
        pvals = anova["PR(>F)"].dropna().rename(lambda x: f"pval_{x}")
        fvals = anova["F"].dropna().rename(lambda x: f"fval_{x}")
        eta = anova["sum_sq"].dropna().apply(lambda x: x/(x+res.ssr)).rename(lambda x: f"eta_p_{x}")
        fit_dict.update(pvals.to_dict())
        fit_dict.update(fvals.to_dict())
        fit_dict.update(eta.to_dict())
    return fit_dict, res
    
def fit_models(data, formulas, features, formula_names=None, feature_names=None, 
               cov_type='HC3', anova_type=2, rank_transform=False):
    if rank_transform:
        data = data.copy()
        data[features] = data[features].rank()
    feature_names = feature_names or [str(x) for x in features]
    formula_names = formula_names or [str(x) for x in formulas]
    all_fits = []
    for feature, feature_name in zip(features, feature_names):
        for formula, formula_name in zip(formulas, formula_names):
            try:
                fit_dict, results = ols_model(data, formula, feature, 
                                              anova_type=anova_type, cov_type=cov_type)
                all_fits.append(dict(fit_dict, model=formula_name, feature=feature_name))
            except Exception:
                pass
        
    fits_df = pd.DataFrame(all_fits)
    return fits_df

def plot_fit(data, feature, formula, x=None, cluster='cluster', ax=None, legend=False,
            print_attr=None, print_pvals=True, **sns_args):
    if not ax:
        fig, ax = plt.subplots()
    # factors = [c for c in data.columns if c in formula]
    # data = data.dropna(subset=factors+[feature])
    out_dict, res = ols_model(data, formula, feature)

    x = x or formula.replace('*','+').split('+')[0].strip()
    sns_args['s'] = sns_args.get('s', 25)
    # if cluster is not None and cluster not in formula:
    #     data[cluster] = data[cluster].fillna('none')
    sns.scatterplot(data=data, y=feature, x=x, hue=cluster, ax=ax, legend=legend, **sns_args)
    
    if cluster is not None and cluster in formula:
        hue = cluster 
        c = None
    else:
        hue = None
        c = 'k'
    y_fit = res.fittedvalues.reindex(data.index)
    sns.lineplot(data=data, y=y_fit, x=x, hue=hue, color=c, legend=False, ax=ax)
    ax.set_xlabel(getattr(x, "label", None) or x)
    ax.set_ylabel(getattr(feature, "label", None) or feature)

    summary = ''
    if print_attr:
        if not isinstance(print_attr, list):
            print_attr = [print_attr]
        for attr in print_attr:
            value = out_dict.get(attr)
            attr_name = getattr(attr, "label", None) or attr
            summary += f"{attr_name} = {value:.2g}\n"
    if print_pvals:
        anova = anova_lm(res, typ=2)
        pvals = anova["PR(>F)"].dropna()
        summary += ", ".join(f"p_{key}={pvals[key]:.2g}" for key in pvals.index)
    ax.text(0.5, 0.99, summary, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='center')
    sns.despine()
    return out_dict

def plot_sig_bars(pvals, pairs_idx, cutoff=0.05, bar=True, label='stars', ax=None, y0=None):
    ax = ax or plt.gca()
    ylim = ax.get_ylim()
    pairs_sig = np.flatnonzero(np.array(pvals)<cutoff)
    
    y0 = y0 or ylim[1]
    n = len(pairs_sig)
    dy = 0.04*(ylim[1]-ylim[0]) # use 4% of y-axis range
    yvals = y0 + dy*np.arange(1, n+1)
    for i, pair in enumerate(pairs_sig):
        plot_sig_bar(pvals[pair], yvals[i], pairs_idx[pair], bar=bar, label=label, ax=ax)

def plot_sig_bar(pval, y, x_pair, bar=True, label='stars', ax=None):
    ax = ax or plt.gca()
    if bar:
        ax.plot(x_pair, [y, y], 'grey')
    if label=='stars':
        text = np.choose(np.searchsorted([1e-3, 1e-2, 5e-2], pval), ['***','**','*',''])
    elif label=='pval':
        text = "p={p:.2}".format(p=pval)
    else:
        text = ''
    ax.annotate(text, xy=(np.mean(x_pair), y), xytext=(0, 2), textcoords='offset points',
        horizontalalignment='center', verticalalignment='center')

def plot_test_bars(data, var, group, test='mannwhitney', group_vals=None, pairs='all', 
                   cutoff=0.05, fdr_method='fdr_bh',
                   label='stars', ax=None, y0=None):
    group_vals = group_vals or data[group].sort_values().unique().tolist()
    pvals, pairs_idx, pairs_names = posthoc_results(data, group, var, group_vals, pairs=pairs, method=test, p_adjust=fdr_method)
    # plot_data = data[ data[group].isin(set.union(*map(set, pairs_list)))]
    plot_data = data[data[group].isin(group_vals)]
    y0 = plot_data[var].max()
    plot_sig_bars(pvals, pairs_idx, cutoff, label=label, ax=ax, y0=y0)
    
def outline_boxplot(ax):
    for i,artist in enumerate(ax.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = artist.get_facecolor()
        if 'col' != 'None':
            artist.set_edgecolor(col)
            artist.set_facecolor('None')

            # Each box has 5(?) associated Line2D objects (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same colour as above
            n=5
            for j in range(i*n,i*n+n):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)
            
def plot_box_cluster_feature(data, y, x='cluster', x_fine=None, label=None, ax=None,
                    palette=None, palette_fine=None, strip_width=0.2, 
                    drop_box=False,
                    test='mannwhitney', pairs='all', cutoff=0.05, fdr_method='fdr_bh',
                    invert_y=False, label_yaxis=False, pad_title=0, title_loc='right',
                    label_counts=True, label_color=False, size=3, highlight=None, 
                    legend=None, **kwargs
                    ):
    data = data.dropna(subset=[x,y]).copy()
    if not hasattr(data[x], 'cat'):
        # make column ordered categorical
        data[x] = data[x].astype('category')
        
    label = label or y
    if ax is None:
        fig, ax = plt.subplots()
    if ax=='gca':
        ax = plt.gca()
    if x_fine:
        subgroups = data.groupby(x)[x_fine].unique()
        x_key = {}
        for i, (subclass, types) in enumerate(subgroups.items()):
            x_key.update({x: i+(j-(len(types)-1)/2)*(2*strip_width/len(types)) for j, x in enumerate(types)})
        data['x_fine'] = data[x_fine].map(x_key)
        # square size to match stripplot behavior
        sns.scatterplot(data=data, x='x_fine', y=y, hue=x_fine, palette=palette_fine,
                      ax=ax, s=size**2, alpha=0.7, legend=legend, **kwargs)
    else:
        sns.stripplot(data=data, x=x, y=y, palette=palette, ax=ax, 
                      jitter=strip_width, s=size, alpha=0.7, **kwargs)
    data_box = data.loc[lambda df: df[x]!=drop_box] if drop_box else data
    sns.boxplot(data=data_box, x=x, y=y, palette=palette, ax=ax, showfliers=False)
    if highlight is not None:
        sns.stripplot(data=data.loc[data.index.intersection(highlight)], 
                      x=x, y=y, palette=palette, ax=ax, jitter=strip_width, size=size*4, marker='X', **kwargs)
    
    outline_boxplot(ax)
    ax.set_xlabel(None)
    sns.despine()
    if label_yaxis:
        ax.set_ylabel(label)
    else:
        ax.set_ylabel(None)
        ax.set_title(label, loc=title_loc, pad=pad_title)
    xlabels = [text.get_text() for text in ax.get_xmajorticklabels()]
    if label_counts:
        counts = data[x].value_counts()
        newlabels = [f"{name}\n(N={counts.loc[name]})" if name!=drop_box
                     else name
                     for name in xlabels]
        newlabels = [name.replace(' ','\\ ') for name in newlabels]
        ax.set_xticklabels(newlabels, rotation=90, ha='center',)
        if label_color:
            for xtick, name in zip(ax.get_xticklabels(), xlabels):
                xtick.set_color(palette[name])
    else:
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    min = data[y].min()
    max = data[y].max()
    if (min>0) and ((min>2) or not ((max<2) and (min>0.4))):
        ax.set_ylim(0, None, auto=True)

    if pairs=='all' and drop_box:
        pairs = list(combinations(set(data[x].unique()).difference(drop_box), 2))
    if pairs is not None:
        plot_test_bars(data, y, x, test=test, group_vals=None, pairs=pairs, fdr_method=fdr_method,
                       ax=ax, cutoff=cutoff, label=None)
        
    min = data[y].min()
    max = data[y].max()
    if (min>0) and ((min>2) or not ((max<2) and (min>0.4))):
        ax.set_ylim(0, None, auto=True)
    if invert_y:
        ax.invert_yaxis()

def plot_boxplot_multiple(data, features, x='cluster', labels=None, horizontal=False, figsize=(4,8),
                          label_yaxis=False, pad_title=0, title_loc='right',
                             plot_function=plot_box_cluster_feature, **kwargs
                            ):
    n = len(features)
    if labels is None:
        labels = features
    elif callable(labels):
        labels = [labels(x) for x in features]
        
    if horizontal:
        fig, axes = plt.subplots(1,n, figsize=figsize, sharex=True)
    else:    
        fig, axes = plt.subplots(n,1, figsize=figsize, sharex=True)
    if n==1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        plot_function(data, y=features[i], x=x, ax=ax, **kwargs)
        if label_yaxis:
            ax.set_ylabel(labels[i])
        else:
            ax.set_ylabel(None)
            ax.set_title(labels[i], loc=title_loc, pad=pad_title)
        
def run_cluster_anova(df, features, cluster_var='cluster', pval='pval_cluster', cov_type='HC3', fdr_method='fdr_bh',):
    df = df.copy()
    df['cluster'] = df[cluster_var]
    
    results = (fit_models(df, ['cluster'], features, cov_type=cov_type).set_index('feature')
        .sort_values('rsquared', ascending=False)
    )
    results[pval] = (
        results[pval].pipe(pd.DataFrame)#in case this is Series
        .fillna(1)
        .apply(lambda col: multipletests(col, method=fdr_method)[1]).astype(float)
    )
    return results

def plot_feature_effect_sizes(results, pval='pval_cluster', val='rsquared', ylabels=None, 
                               figsize=(1.5,8), nshow=20, sort=True):
    if sort:
        results = results.sort_values(val, ascending=False)
    data = results.loc[:,pval]
    stars = pd.cut(data.iloc[:nshow], [0, 0.001, 0.01, 0.05, 1], labels=['***','**','*',''])


    fig, ax = plt.subplots(figsize=figsize)
    bardata = results.iloc[:nshow].loc[:,val].reset_index().melt(id_vars=['feature'])
    sns.barplot(data=bardata, y='feature', x='value', hue='variable')
    sns.despine()
    if ylabels is not None:
        ax.set_yticklabels([ylabels[label.get_text()] for label in ax.get_yticklabels()])
    ax.set_ylabel(None)
    ax.set_xlabel(val)
    # ax.set_title('Cluster ANOVA $\eta^2$')
    ax.get_legend().remove()

    nfeat = min(nshow, len(ylabels)) if ylabels is not None else nshow
    for i, p in enumerate(ax.patches):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        annot = stars[i]
        if not annot:
            col = p.get_facecolor()
            p.set_edgecolor(col)
            p.set_facecolor('None')
        else:
            space = 0.005
            _x = p.get_x() + p.get_width() + float(space)
            _y = p.get_y() + p.get_height()
            ax.text(_x, _y, annot, ha="left")

from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as skp

def posthoc_results(data, group_col, val_col, group_vals=None, p_adjust=None, cutoff=0.05,
                 method='dunn', pairs='all'):
    fcn = getattr(skp, f"posthoc_{method}")
    if group_vals is not None:
        data = data[data[group_col].isin(group_vals)]
    else:
        group_vals = data[group_col].sort_values().unique().tolist()
    # groups = data.dropna(subset=[val_col, group_col]).groupby(group_col)[val_col].apply(list)
    pvals_df = fcn(data.dropna(subset=[val_col, group_col]),
                   val_col, group_col, p_adjust=None)
    stacked = pvals_df.stack()
    if pairs=='all':
        pairs = list(combinations(group_vals, 2))
        # indices = np.triu_indices(len(group_vals), k=1)
        # pvals = pvals_df.values[indices]
        # index = [*zip(*indices)]
    pvals = stacked[pairs].values
    nums = {name: i for i, name in enumerate(group_vals)}
    index = [(nums[pair[0]], nums[pair[1]]) for pair in pairs]
    if p_adjust is not None:
        pvals = multipletests(pvals, method=p_adjust)[1]
    pairs = pd.DataFrame(dict(pvals=pvals, ind=index, names=pairs))
    sig = pairs.loc[lambda df: df.pvals<cutoff]
    # sig_list = [' - '.join(group_vals[i] for i in pair) for pair in sig_pairs.index]
    return sig.pvals.values, sig.ind.values, sig.names.values

def run_kw_dunn(data, features, group_col, fdr_method='fdr_bh', posthoc='dunn', cutoff=0.05):
#     features = [x for x in features if data.groupby(group_col)[x].var().min() > 0]
    records = []
    pval = 'pval'
    for f in features:
        df = data.dropna(subset=[f, group_col])
        groups = df.groupby(group_col, observed=True)[f].apply(list)
        k = len(groups)
        n = len(df)
        rec = dict(feature=f)
        rec['KW_H'], rec[pval] = kruskal(*groups.values)
        # rec['rsquared'] = (rec['KW_H'] - k + 1)/(n - k)
        rec['epsilon2'] = rec['KW_H']/(n-1)
        records.append(rec)
    results = pd.DataFrame.from_records(records, index='feature')

    if fdr_method is not None:
        results['pval_fdr'] = results[pval].pipe(lambda col: multipletests(col, method=fdr_method)[1])
    if posthoc:
        results = results.assign(pairs=None, ipairs=None, pair_count=None)
        for f in features:
            df = data.dropna(subset=[f, group_col])
            if results.loc[f,'pval_fdr'] < cutoff:
                pvals, index, names = posthoc_results(df, group_col, f, p_adjust='fdr_bh')
                str_index = [''.join(str(i) for i in pair) for pair in index]
                results.loc[f, ['pairs', 'ipairs', 'pair_count']] = names, str_index, len(pvals)
    return results.sort_values(pval)

def run_anova_pairs(mouse_df, human_df, features, cluster='cluster', cov_type='HC3', fdr_method='fdr_bh',):
    pval = f'pval_{cluster}'
    label = 'rsquared'
    fdr_method='fdr_bh'
    
    results = (pd.concat(
        [
        fit_models(human_df, [cluster], features, cov_type=cov_type).set_index('feature'),
        fit_models(mouse_df, [cluster], features, cov_type=cov_type).set_index('feature'),
        ],
        keys=[
            'human',
            'mouse',
             ],
        axis=1)
        .sort_values(('human','rsquared'), ascending=False)
        .swaplevel(axis=1)
    )
    results = results.dropna(subset=[(pval,'human')])
    if fdr_method is not None:
        results[pval] = (
            results[pval].pipe(pd.DataFrame)#in case this is Series
            .fillna(1)
            .apply(lambda col: multipletests(col, method=fdr_method)[1]).astype(float)
        )
    return results

def select_distinct(ranked, corr, nfeat=10, threshold=0.8):
    distinct_features = []
    for x in ranked:
        if len(distinct_features)==nfeat:
            return distinct_features
        for y in distinct_features:
            if np.abs(corr.loc[x,y]) > threshold:
                break
        else:
            distinct_features.append(x)
    return distinct_features

def compile_feature_effects_comparison(results_list, group_list, rank='max', effect='rsquared', pval='pval_cluster',
                                        corr_filter=None, nfeat=10, corr_thresh=0.8):
    results = pd.concat(results_list, keys=group_list, axis=1).swaplevel(axis=1)
    results[rank] = results[effect].agg(rank, axis=1)
    results['diff'] = results[(effect,group_list[0])] - results[(effect,group_list[1])]
    ranked_features = results.sort_values(rank, ascending=False).index
    if corr_filter is not None:
        ranked_features = select_distinct(ranked_features, corr_filter, nfeat=nfeat, threshold=corr_thresh)
    else:
        ranked_features = ranked_features[:nfeat]
    return results.loc[ranked_features].sort_values('diff', ascending=False)

def plot_feature_effects_comparison_barplot(results, pval='pval_cluster', val='rsquared', ylabels=None,
                                            figsize=(1.5,8), palette=None, nshow=None):
    fig, ax = plt.subplots(figsize=figsize)
    results.index.name = 'feature'
    data = results.loc[:,pval]
    stars = data.apply(lambda x: pd.cut(x, [0, 0.001, 0.01, 0.05, 1], labels=['***','**','*',''])).astype(str).values

    bardata = results.loc[:,val].reset_index().melt(id_vars=['feature'])

    sns.barplot(data=bardata, y='feature', x='value', hue='variable', palette=palette)
    sns.despine()
    if ylabels is not None:
        ax.set_yticklabels([ylabels[label.get_text()] for label in ax.get_yticklabels()])
    ax.set_ylabel(None)
    ax.set_xlabel('$\eta^2$')
    ax.get_legend().remove()

    nfeat = nshow or len(results)
    for i, p in enumerate(ax.patches):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        ifeat = i % nfeat
        ispecies = i // nfeat
        annot = stars[ifeat, ispecies]
        if not annot:
            col = p.get_facecolor()
            p.set_edgecolor(col)
            p.set_facecolor('None')
        else:
            space = 0.005
            _x = p.get_x() + p.get_width() + float(space)
            _y = p.get_y() + p.get_height()
    #         ax.text(_x, _y, annot, ha="left")
    
from . import classification as ac
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_cv_score(pipeline, X, y, scoring=None, cv=RepeatedStratifiedKFold()):
    scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv)
    print("CV accuracy: {:.2f}+/-{:.2f}".format(np.mean(scores), np.std(scores)))

def plot_cv_cm(pipeline, X, y, scoring=None, cv=RepeatedStratifiedKFold(), labels=None, figsize=(4,4), 
            cmap='inferno', cbar_position='right'):
    cm = ac.cv_confusion_matrix(X, y, pipeline, cv, normalize_axis=1)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=cmap)
    n_classes = len(labels)
    ax.set(xticks=np.arange(n_classes),
                yticks=np.arange(n_classes),
                xticklabels=labels,
                yticklabels=labels,
                ylabel="True label",
                xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    divider = make_axes_locatable(ax)
    orientation = 'horizontal' if cbar_position == 'top' else 'vertical'
    cax = divider.append_axes(cbar_position, size="10%", pad=0.1)
    cb = plt.colorbar(im, cax=cax,  orientation=orientation)
    if cbar_position == 'top':
        cax.xaxis.set_label_position('top')
        cax.xaxis.tick_top()
    cb.set_label("Fraction of row")
    return cm

def run_regressions(df, features, variable, pval=None, cov_type='HC3', fdr_method='fdr_bh', pretty_format=False):
    
    results = (fit_models(df, [variable], features, cov_type=cov_type).set_index('feature')
        .sort_values('rsquared', ascending=False)
    )
    if pval is None:
        pval = [col for col in results if col.startswith('pval_')]
    results = results.dropna(subset=pval)
    for x in pval:
        results[f"{x}_fdr"] = (
            results[x].pipe(pd.DataFrame)#in case this is Series
            .apply(lambda col: multipletests(col, method=fdr_method)[1]).astype(float)
    )
    if pretty_format:
        cols = [
            'nobs',
            f"pval_{variable}",
            'rsquared'
        ]
        names = [
            'number cells',
            'p-value (FDR-BH)',
            'Rsquared'
        ]
        results = (results[cols]
                .rename(columns=dict(zip(cols, names)))
            )
    return results

import scipy.stats as stats
def run_twosamp(df, features, variable, fdr_method='fdr_bh', sort_by='pval_mw'):
    records = []
    for feature in features:
        grouped = df.dropna(subset=[variable, feature]).groupby(variable)
        subsets = [group[feature] for val, group in grouped]
        assert len(subsets) == 2
        nums = grouped.size().values
        u, pval_mw = stats.mannwhitneyu(subsets[0], subsets[1])
        roc_auc = u/(nums[0]*nums[1])
        roc_auc = roc_auc if roc_auc > 0.5 else 1 - roc_auc
        mw_r = 2*roc_auc - 1
        t, pval_t = stats.ttest_ind(subsets[0], subsets[1])
        cohens_d = np.abs(np.mean(subsets[0]) - np.mean(subsets[1])) / np.std(df[feature])
        records.append(dict(pval_mw=pval_mw, mw_r=mw_r, roc_auc=roc_auc, 
                            pval_t=pval_t, cohens_d=cohens_d,
                            feature=feature, nobs=nums[0]+nums[1]))
        
    results = pd.DataFrame(records).sort_values(sort_by, ascending=True).set_index('feature')
    results["pval_t_fdr"] = multipletests(results["pval_t"], method=fdr_method)[1]
    results["pval_mw_fdr"] = multipletests(results["pval_mw"], method=fdr_method)[1]
    return results

        
def permutation_test(data, dep_vars, effect_fcn, bs_agg_fcn=np.max, n_permutations=1000,):
    effects = effect_fcn(data, dep_vars)
    data = data.copy()

    effects_perm = np.zeros(n_permutations)

    for p in range(n_permutations):
        data[dep_vars] = data.loc[np.random.permutation(data.index), dep_vars].values
        effects_perm[p] = bs_agg_fcn(effect_fcn(data, dep_vars))

    """Compute p-values for each cluster. Make a vector indicating which
    variables are significant."""
    pvals = [(np.sum(effects_perm > x) + 1) / (n_permutations + 1) 
             for x in effects]

    results = effects.to_frame().assign(pval_corrected=pvals)
    return results

from scipy.stats import spearmanr, pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.isotonic import IsotonicRegression
def plot_spearman(data, x, y, smooth=True, hue=None, palette=None, label=None, stats=True, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    corr, p = spearmanr(data[x], data[y], nan_policy='omit')
    sns.scatterplot(data=data, x=x, y=y, hue=hue, legend=False, ax=ax, palette=palette, **kwargs)

    if smooth:
        smoothed = lowess(data[y], data[x])
        ax.plot(*smoothed.T, 'grey')
    else:        
        xsmooth = np.linspace(min(data[x]), max(data[x]), 200)
        ysmooth = IsotonicRegression(increasing='auto').fit(data[x], data[y]).predict(xsmooth)
        ax.plot(xsmooth, ysmooth, 'grey')

    summary = f"ρ={corr:.2g}, p={p:.2g}"
    if stats:
        ax.text(0.5, 0.99, summary, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='center')
    sns.despine()

def run_correlations(data, features, xvar, fdr_method='fdr_bh'):
    results = []
    for gene in features:
        if gene not in data or data[gene].pipe(lambda x: sum(x>1)) < 5:
            continue
        df = data.dropna(subset=[gene, xvar])
        r, p = spearmanr(df[gene], df[xvar])
        out = {
        "feature":gene,
        # "feature":xvar,
        'r':r,
        'pval':p,
        }
        results.append(out)
    results = pd.DataFrame.from_records(results, index='feature')
    if fdr_method is not None:
        results['pval_fdr'] = results['pval'].pipe(lambda col: multipletests(col, method=fdr_method)[1])
    return results.sort_values('pval')

def run_species_subclass_stats(data, features, compare='species', group_var='homology_type', groups=None, rank_transform=False,
                               cutoff=0.05, compare_first=True, fdr_anova='fdr_bh', fdr_subgroup=None, anova_type=1):
    if groups is None:
        groups = data[group_var].unique()
    model = f"{compare}*{group_var}" if compare_first else f"{group_var}*{compare}"
    results = fit_models(data, [model], features, 
                         rank_transform=rank_transform, anova_type=anova_type);
    pvals = [x for x in results.columns if 'pval' in x]
    results[pvals] = (results[pvals]
            .apply(lambda col: multipletests(col, method=fdr_anova)[1]).astype(float))
    results = results.sort_values(f'eta_p_{compare}', ascending=False).set_index('feature')
    results['interaction'] = results[f"pval_{model.replace('*',':')}"] < cutoff
    
    cols = [f'eta_p_{compare}', f'pval_{compare}', f"pval_{model.replace('*',':')}", 'rsquared', 'interaction']
    
    interaction_features = results.loc[results.interaction].index
    if len(interaction_features)>0:
        interaction_results = subgroup_comparisons(data, interaction_features, group_var, compare, 
                                               groups=groups, cutoff=cutoff, fdr_method=fdr_subgroup)
        return results[cols].join(interaction_results)
    else:
        return results[cols]

def subgroup_comparisons(df, features, group_var, compare, groups=None, fdr_method=None, cutoff=0.05):
    if groups is None:
        groups = df[group_var].unique()
    records = []
    for feature in features:
        pvals = {}
        stat_results = {}
        for group in groups:
            data = df.loc[df[group_var]==group]
            grouped = data.dropna(subset=[compare, feature]).groupby(compare)
            subsets = [x[feature] for _, x in grouped]
            if len(subsets) == 2:
                u, pval_mw = stats.mannwhitneyu(*subsets)
                pvals[f"{group}"] = pval_mw
                m, n = map(len, subsets)
                rho = u/(m*n)
                stat_results[f"auc_{group}"] = rho
                stat_results[f"md_auc_{group}"] = mwu_mde_rho(m, n)
        pval_vals = np.array(list(pvals.values()))
        if fdr_method is not None:
            pval_vals = multipletests(pval_vals, method=fdr_method)[1]
            pvals = dict(zip(pvals.keys(), pval_vals))
        sig_groups = ", ".join(np.array(list(pvals.keys()))[pval_vals < cutoff])
        records.append(dict(feature=feature, sig_groups=sig_groups, **pvals, **stat_results))
    results = pd.DataFrame.from_records(records).set_index('feature')
    return results

def df_fisher(df, cluster, meta, cluster_name, test=stats.fisher_exact):
    ct = pd.crosstab(df[meta], df[cluster]==cluster_name)
    out = test(ct)
    
    return out.pvalue if hasattr(out, 'pvalue') else out[1]

def fisher_test_all(df, cluster, meta, test=stats.fisher_exact, fdr_method='fdr_bh'):
    df = df.dropna(subset=[cluster])
    names = df[cluster].unique()
    results = pd.Series({cluster_name: df_fisher(df, cluster, meta, cluster_name, test) 
           for cluster_name in names}).sort_values()
    if fdr_method is not None:
        results = pd.Series(multipletests(results, method=fdr_method)[1], index=results.index)
    return results

from scipy.optimize import root_scalar
# avoid recalculating MWU PMF values
from scipy.stats._mannwhitneyu import _MWU as MWU
mwu = MWU()

def mwu_critical_u(m, n, p_cutoff=0.05, two_sided=True):
    pmfs = mwu.pmf(np.arange(0, m*n//2), m, n)
    cdfs = np.cumsum(pmfs)
    if two_sided: p_cutoff /= 2
    return np.flatnonzero(cdfs > p_cutoff)[0] - 1

def mwu_power(m, n, rho):
    U = mwu_critical_u(m, n)
    rho = min(rho, 1-rho)
    return stats.binom.cdf(U, m*n, rho)

def mwu_mde_rho(m, n, power=0.8):
    if mwu_power(m, n, 1) < power:
        return 1
    f = lambda rho: mwu_power(m, n, rho) - power
    res = root_scalar(f, bracket=(0.5, 1))
    return res.root