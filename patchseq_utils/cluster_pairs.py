import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut
from .pipelines import preproc_pipeline
from .util import short_feat

def dprime(y_true, y_score):
    area = roc_auc_score(y_true, y_score, average=None, multi_class='ovo')
    if np.isclose(area, 1):
        n = len(y_true)
        area = (n-1)/n
    return np.sqrt(2) * norm.ppf(area)

def dprime_gaussian(y_true, y_score):
    x2 = y_score[y_true==0]
    x1 = y_score[y_true==1]
    x1_mean = np.mean(x1)
    x1_var = np.var(x1)
    x2_mean = np.mean(x2)
    x2_var = np.var(x2)
    dprime = ((x1_mean - x2_mean) /
        np.sqrt(0.5 * (x1_var + x2_var)))
    return dprime
    

dprime_feat = lambda clf: (np.abs(clf.theta_[0] - clf.theta_[1]) / 
                           np.sqrt(np.sum(clf.var_, axis=0)))

def pairwise_cluster_distances(clf, data, features, cluster_label, 
                              fit_pairwise=True, details=False, ranking=dprime_feat,
                               metric=dprime, method='predict_proba', cv=None):
    X0 = data[features].values
    X = preproc_pipeline().fit_transform(X0)
    types = data[cluster_label].astype('category')
    types = types.cat.remove_unused_categories()
    y = types.cat.codes
    cluster_order = types.cat.categories
    n_clust = len(cluster_order)
    
#     flat prior compares likelihood rather than posterior
    if not fit_pairwise:
#         clf = clf(priors=np.ones(n_clust)/n_clust if flat_prior else None)
        y_pred = cross_val_predict(clf, X, y, method='predict_proba')
#     else:
#         clf = clf(priors=np.ones(2)/2 if flat_prior else None)
        
    distances = np.zeros((n_clust, n_clust))
    records = list()
    for i in range(1, n_clust):
        for j in range(i):
            subset = data[cluster_label].isin(cluster_order[[i,j]])
            if fit_pairwise:
                y_i = y[subset]==i
                if cv==False:
                    clf.fit(X[subset, :], y_i)
                    decision_vars = getattr(clf, method)(X[subset, :])
                else:
                    decision_vars = cross_val_predict(clf, X[subset, :], y_i, method=method, cv=cv)
                y_pred = decision_vars[:,1]
#                 y_pred = decision_vars[:,1] - decision_vars[:,0]
                distances[i,j] = metric(y_i, y_pred)
                
                if details:
                    clf.fit(X[subset, :], y_i)
                    importance = ranking(clf)
                    i_feat = np.argmax(importance)
                    records.append({
                        'feature': features[i_feat],
                        'importance': importance[i_feat],
                        'cluster_1': cluster_order[i],
                        'cluster_2': cluster_order[j],
                    })
                
            else:
#         won't work because of roc_auc multiclass limitation
                distances[i,j] = metric(y[subset], clf.predict_proba(X[subset, :]))[i]
    
#     distances[np.isnan(distances)] = 0
    distances = distances + distances.T
    ephys_dprime = pd.DataFrame(distances, index=cluster_order, columns=cluster_order)
    return ephys_dprime, records


def ova_cluster_distances(clf, data, features, cluster_label, n_feat=2,
                         metric=dprime):
    X0 = data[features].values
    X = preproc_pipeline().fit_transform(X0)
    types = data[cluster_label].astype('category')
    y = types.cat.codes
    cluster_order = types.cat.categories
    n_clust = len(cluster_order)
        
    distances = np.ones(n_clust) * np.nan
    records = list()
    for i in range(n_clust):
        y_i = y==i
        y_pred = cross_val_predict(clf, X, y_i, method='predict_proba')[:,1]
        distances[i] = metric(y_i, y_pred)

        clf.fit(X, y_i)
        dprime_feat = np.abs(clf.theta_[0] - clf.theta_[1])/np.sqrt(np.sum(clf.var_, axis=0))
        feat_order = np.argsort(dprime_feat)[::-1]
        for j in range(n_feat):
            records.append({
                'feature': features[feat_order[j]],
                'dprime': dprime_feat[feat_order[j]],
                'cluster': cluster_order[i],
                'rank':j,
            })

    return distances, records


from scipy.cluster import hierarchy
import seaborn as sns
import matplotlib.pyplot as plt
    
def plot_dprime(clf, data, features, cluster='t-type',  cluster_list=None, 
                metric=dprime, method='predict_proba', cv=None):
    dprime, _ = pairwise_cluster_distances(clf, data, features, cluster, 
                                    fit_pairwise=True, details=False, 
                                           metric=metric, method=method, cv=cv)
    if cluster_list is not None:
        dprime = dprime.reindex(index=cluster_list, columns=cluster_list)
    ax = sns.heatmap(dprime, cmap='rocket_r', vmin=0, vmax=3, cbar=True)
    if dprime.isna().any(axis=None):
        ax.patch.set_edgecolor('lightgrey')
        ax.patch.set_hatch('//')
    plt.axis('equal')
    # ax.set_frame_on(True)
    [s.set_visible(False) for s in ax.spines.values()]
    plt.show()
    return dprime

def plot_dprime_tree(clf, data, features, cluster,**kwargs):
    dist, _ = pairwise_cluster_distances(clf, data, features, cluster, 
                                    fit_pairwise=True, details=False)
    y = dist.T.values[np.triu_indices_from(dist, 1)]
    y[y>5] = 5
    y[y<0] = 0
    Z = hierarchy.linkage(y, **kwargs)
    fig, ax = plt.subplots(figsize=(6,4))
    hierarchy.dendrogram(Z, ax=ax, labels=clusters, orientation='right')
    
def plot_dprime_features(dprime, records):
    clusters = dprime.index
    df = pd.DataFrame.from_records(records, index=['cluster_1', 'cluster_2'])
    df['short_feat'] = df['feature'].map(short_feat)
    labels = (df.reset_index().pivot(index='cluster_1', columns='cluster_2', values='short_feat')
              .reindex(index=clusters, columns=clusters)
              .fillna(''))

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(dprime, annot=labels, fmt='s', cmap='rocket_r',
                annot_kws=dict(size=10), cbar_kws=dict(label="d'"), vmin=0, vmax=3,
               cbar=True, ax=ax)