from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import sklearn.metrics as metrics
from itertools import combinations
import numpy as np
import seaborn as sns

class OutlierExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs ):
        self.kwargs = kwargs

    def transform(self, X):
        lof = LocalOutlierFactor(**self.kwargs)
        count = 0
        for i in range(X.shape[1]):
            labels = lof.fit_predict(X[:,[i]])
#             want fixed threshold not percentile here
#             nof = lof.negative_outlier_factor_
#             outliers = nof > np.quantile(nof, 0.95)
#             outliers = (labels==-1)
            outliers = (lof.negative_outlier_factor_ < -20)
            count += outliers.sum()
            X[outliers, i] = np.nan
#         print(f"{count} outlier values removed")
        return X

    def fit(self, X, y=None):
        return self
    
def preproc_pipeline():
    return Pipeline(steps=[
    ('norm', RobustScaler()),
    ('impute1', KNNImputer()),
    ('outliers', OutlierExtraction()),
    ('impute2', KNNImputer()),
])


from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def plot_cm(clf, data, features, cluster_label, cv=None, scoring=None, **kwargs):
    X = data[features].values
    types = data[cluster_label].astype('category')
    y = types.cat.codes
    pipeline = Pipeline(steps=[
        ('preproc', preproc_pipeline()),
        ('class', clf)
    ])
    if cv==False:
        pipeline.fit(X, y)
        print("Accuracy: {:.2f}".format(pipeline.score(X,y)))
        cm = metrics.confusion_matrix(y, pipeline.predict(X), normalize='true')
    else:
        cv = cv or RepeatedStratifiedKFold()
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv)
        print("CV accuracy: {:.2f}+/-{:.2f}".format(np.mean(scores), np.std(scores)))
        cm = cv_confusion_matrix(X, y.values, pipeline, cv, normalize_axis=1)

    cmplot = metrics.ConfusionMatrixDisplay(cm, display_labels=types.cat.categories)
    fig, ax = plt.subplots(figsize=(5,5))
    cmplot.plot(ax=ax, **kwargs)
    cmplot.im_.set_clim(0, 0.9)
    ax.set_xticklabels(ax.get_xmajorticklabels(), rotation=45, fontstyle='italic', ha='right')
    


def inter_intra_mean_ratio(data, labels, metric='euclidean'):
    """Clustering metric comparing mean pairwise distances, inter- vs intra-cluster
    
    Parameters
    ----------
    data : 2d array
        samples x features
    labels : 1d array
        cluster labels for each sample
    metric: string (default='euclidean')
        distance metric to use
    Returns
    -------
    float
        inter/intra distance ratio
    """    
    dist = metrics.pairwise_distances(data, metric=metric)
    intra = np.mean([
        np.mean([dist[x,y] for x,y in combinations(np.flatnonzero(labels==i),2)]) 
        for i in np.unique(labels)])
    inter = np.mean([
        np.mean(dist[np.ix_(labels==i, labels==j)]) 
        for i,j in combinations(np.unique(labels),2)])
    return inter/intra

def cv_confusion_matrix(X, y, classifier, cv, normalize_axis=None):
    output = []
    for train, test in cv.split(X, y):
        classifier.fit(X[train,:], y[train])
        output.append(metrics.confusion_matrix(y[test], classifier.predict(X[test])))
    cm = np.mean(output, axis=0)
    if normalize_axis is not None:
        cm = cm / cm.sum(axis=normalize_axis, keepdims=True)
    else:
        cm = cm / cm.sum()
    return cm

from umap import UMAP
from sklearn.decomposition import SparsePCA
def calc_transforms(data, features, **kwargs):
    data = data.loc[lambda df: df[features].notna().sum(axis=1) > 0].copy()
    pipeline = Pipeline(steps=[
        ('norm', PowerTransformer()),
        ('impute', KNNImputer()),
    ])
    Y = pipeline.fit_transform(data[features])
#     norm_features = ["norm_"+x for x in features]
#     data = data.assign(**dict(zip(norm_features, Y.T)))
    
    transforms = {'umap': UMAP, 'spca': SparsePCA}
    for name, fcn in transforms.items():
        obj = fcn(**kwargs.get(name)) if name in kwargs else fcn()
        Z = obj.fit_transform(Y)
        data = data.assign(**{f"{name}0": Z[:,0], f"{name}1": Z[:,1]})
    return data

def plot_transform(data, var, name="umap", legend=False, **kwargs):
    plt.figure(figsize=(4,4))
    xy = dict(x=f"{name}0", y=f"{name}1")
    if data[var].isna().sum()>0:
        sns.scatterplot(data=data.loc[data[var].isna()], color='grey', **xy, **kwargs)
    sns.scatterplot(data=data, hue=var, legend=legend, **xy, **kwargs)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=var)
    sns.despine()
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    name = name.upper()
    plt.xlabel(f'{name}-1')
    plt.ylabel(f'{name}-2')
    plt.title(name)