"""Perform proper orthogonal decomposition for omics timeseries data."""

import numpy as np
from scipy import linalg
import pandas
import matplotlib
from matplotlib import pyplot

class POD(object):
    """POD computations and evaluation of results.
"""
    def __init__(self, ds, features=None, conditions=None, cluster=None, cluster_components=1):
        """The constructor runs a POD on a given dataset.

Parameters
---
ds : Dataset
    `Dataset` object on which to do the POD computation.
features : list of str, default=None
    Features to include in the analysis, default all.
conditions : list of str, default=None
    Conditions to include in the analysis, default all.
cluster : object, default=None
    Pass an instance of one of the classes in [sklearn.cluster](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) to perform
    clustering on the features.
    Clustering will be done in the singular vector (principal component) space.
cluster_components : int, default=1
    Dimensions of the singular vector space to use for clustering. For example, if `cluster_components=2`, clustering will be done based
    on the first two singular vectors (principal components).

Attributes
---
ds : Dataset
    Reference to the `Dataset` object which was given to constructor.
features : list of str
    Features used in the POD. Can be a subset of the features in the original dataset.
condition_list : list of str or None
    List of conditions used in the POD. Can be a subset of the conditions in the original dataset.
sing_values : array
    One-dimensional array of singular values resulting from the POD. 
    The length of the array is either the number of considered features or the number of considered samples, depending on what is smaller.
sample_weights : array
    Two-dimensional array of sample weights in the POD.
    The array shape is equal to the number of considered samples by the number of singular values.
    This is equal to the left singular vectors of the considered data matrix.
feature_weights : array
    Two-dimensional array of feature weights in the POD.
    The array shape is equal to the number of considered features by the number of singular values.
    This is equal to the right singular vectors of the considered data matrix.    
cluster : object
    Instance of the sklearn class which was used for clustering.
cluster_components : int, default=1
    Number of singular vectors / principal components, in decreasing order of weight, on which to do the clustering.
"""
        if features is None:
            self.features = ds.features
        else:
            for f in features:
                if f not in ds.features:
                    raise ValueError(f'Feature "{f}" not found in dataset features: {ds.features}')
                self.features = features
        if conditions is None:
            self.condition_list = ds.condition_list
        else:
            for c in conditions:
                if c not in ds.condition_list:
                    raise ValueError(f'Condition "{c}" not found in dataset condition list: {ds.condition_list}')
            self.condition_list = conditions
        self.ds = ds

        self.sample_weights, self.sing_values, Vt = linalg.svd(np.asarray(self.ds.data[self.features]))
        self.feature_weights = Vt.T

        self.cluster = cluster
        if self.cluster is not None:
            self.labels = self.cluster.fit_predict(self.feature_weights[:cluster_components, :])

            
    def plot_singular_values(self, ax=None, trafo=None, fit_line=False, line_plot_args={}, **kwargs):
        """Plot the singular values.
"""
        if ax is None:
            fig, ax = pyplot.subplots(1, 1)
            ax.set_ylabel("Singular values")
            ax.set_xlabel("Component index")
            new_fig = True
        else:
            new_fig = False
        if trafo is None:
            values = self.sing_values
        else:
            values = trafo(self.sing_values)
        ax.plot(self.sing_values, **kwargs)
        if fit_line is not False:
            if fit_line is True:
                M = np.ones(shape=(self.sing_values.shape[0], 2))
                M[:,1] = np.arange(self.sing_values.shape[0])
                fit_vals = values
            elif fit_line < 1:
                num_fit = np.round(fit_line / self.sing_values.shape[0])
                fit_vals = values[-num_fit:]
            else:
                fit_vals = values[-fit_line:]
            fit_param = linalg.lstsq(M[-last_n:,:], np.log(S[-last_n:]), rcond=None)[0]
            ax.plot(M[:,1], M.dot(fit_param), **line_plot_args)
        

