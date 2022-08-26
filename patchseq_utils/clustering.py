
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def get_cluster_pipeline():
    cache_dir = "/local1/storage/temp/"
    cache_dir = None
    pipe = make_pipeline(LabeledGMM(n_components=2, init_params='kmeans'), memory=cache_dir)
    return pipe

# could pipeline this all and extract intermediate?
def add_norm_features_and_impute(data, features):
    pipeline = Pipeline(steps=[
        ('norm', PowerTransformer()),
        ('impute', KNNImputer()),
    ])
    Y = pipeline.fit_transform(data[features])
    norm_features = ["norm_"+x for x in features]
    data = data.assign(**dict(zip(norm_features, Y.T)))
    norm = pipeline.named_steps['norm']
    # data[features] = norm.inverse_transform(Y)
    return data, norm_features
    
def weighted_f1(y_true, y_pred):
    beta = 2
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    return (1+beta) * (precision * recall) / (beta*precision + recall)
    
def score_feature_set_gmm(pipe, data, features, cluster_var, cluster_name, scorers, seed=0):
    y_true = data[cluster_var]==cluster_name
    np.random.seed(seed)
    y_pred = pipe.fit_predict(data[features])
    i_true = int(np.mean(y_pred[y_true]) > 0.5)
    return {sc.__name__: sc(y_true, y_pred==i_true) for sc in scorers}

def plot_feature_set_gmm(pipe, data, features, cluster_var, cluster_name, feature_names=None, palette=None, ax=None):
    y_true = data[cluster_var]==cluster_name
    gmm = pipe
    gmm.fit(data[features], y_true)
    plot_2d_classifier(pipe, data, y_true, features, cluster_name, feature_names, palette, ax)
    
def plot_2d_classifier(pipe, data, y_true, features, cluster_name, feature_names=None, palette=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    sns.scatterplot(x=features[0], y=features[1], data=data, hue=y_true, ax=ax, legend=False, 
                    palette={False: 'black', True: 'orange' if not palette else palette[cluster_name]})
    if feature_names is not None:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel(features[0].replace('norm_','').replace('_hero',''))
        ax.set_ylabel(features[1].replace('norm_','').replace('_hero',''))
    ax.set_title(cluster_name)
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    plot_prob(ax, pipe)
    
    
def plot_prob(ax, gmm):
    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = gmm.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z/np.sum(Z, axis=1)
    zz = Z[:, 1].reshape(xx.shape)
    plt.sca(ax)
    img = ax.pcolormesh(xx, yy, zz, cmap="RdBu_r", zorder=0, shading='auto', vmin=0, vmax=1)
    ax.contour(xx, yy, zz, [0.5], linewidths=2.0, colors="white")
    return img
    
def plot_ellipses(ax, gmm):
    for n in range(gmm.means_.shape[0]):
        covariances =  gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], 180 + angle,
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
    ax.set_aspect("equal", "datalim")

from itertools import combinations
def cluster_combinations_gmm(pipe, data, features, cluster_var, cluster_name, n_repeat=1):
    scorers = [metrics.f1_score, metrics.accuracy_score, weighted_f1]
    records = []
    for i in range(n_repeat):
        for feature in features:
            scores = score_feature_set_gmm(pipe, data, [feature], cluster_var, cluster_name, scorers, seed=i)
            records.append(dict(features=feature, **scores))

        for features in combinations(features, 2):
            scores = score_feature_set_gmm(pipe, data, list(features), cluster_var, cluster_name, scorers, seed=i)
            records.append(dict(features=', '.join(features), **scores))

    df = pd.DataFrame.from_records(records)
    return df.sort_values('f1_score', ascending=False)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from joblib import Parallel, delayed
from sklearn.multiclass import _fit_binary
from sklearn.base import clone

# class PrefitOvr(OneVsRestClassifier):
#     def __init__(self, estimators):
#         self.estimators = estimators

#     def fit(self, X, y):
#         return self
    
class CustomOneVsRestClassifier(OneVsRestClassifier):

    # Changed the estimator to estimators which can take a list now
    def __init__(self, estimators, n_jobs=1):
        self.estimators = estimators
        # self.estimator = clone(estimators[0])
        self.n_jobs = n_jobs

    def fit(self, X, y):
        
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)

        # This is where we change the training method
        # self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
        #     estimator, X, column, classes=[
        #         "not %s" % self.label_binarizer_.classes_[i],
        #         self.label_binarizer_.classes_[i]])
        #     for i, (column, estimator) in enumerate(zip(columns, self.estimators)))
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            estimator, X, column)
            for i, (column, estimator) in enumerate(zip(columns, self.estimators)))
        return self
    
    
class LabeledGMM(GaussianMixture):
    # def __init__(self):
    #     super().__init__()
    def fit(self, X, y):
        super().fit(X, y)
#         this is necessary to calibrate
        self.classes_ = range(len(self.weights_))
        y_pred = self.predict(X)
        if np.mean(y_pred[y]) < 0.5:
            self.flip_classes()
        return self
    def fit_predict(self, X, y):
        y_pred = super().fit_predict(X, y)
        if np.mean(y_pred[y]) < 0.5:
            self.flip_classes()
        return y_pred
    def flip_classes(self):
        self.weights_ = self.weights_[::-1]
        self.means_ = self.means_[::-1, :]
        self.covariances_ = self.covariances_[::-1, :, :]
        self.precisions_ = self.precisions_[::-1, :, :]
        self.precisions_cholesky_ = self.precisions_cholesky_[::-1, :, :]
            
class PandasPipeline(Pipeline):
    def transform(self, X):
        Y = super().transform(X)
        return pd.DataFrame(Y, columns=self.get_feature_names_out())
    def fit_transform(self, X, y):
        Y = super().fit_transform(X, y)
        return pd.DataFrame(Y, columns=self.get_feature_names_out())