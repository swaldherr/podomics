"""Perform proper orthogonal decomposition for omics timeseries data."""

import numpy as np
from scipy import linalg
import pandas
import matplotlib
from matplotlib import pyplot

from .dataset import Dataset

class POD(object):
    """POD computations and evaluation of results.
"""
    def __init__(self, ds, features=None, conditions=None, cluster=None, cluster_components=1, **kwargs):
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
    Internal `Dataset` object containing only the conditions / features that were used in the POD.
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

        df = ds.data.query(f"{ds.condition} in @self.condition_list") # TODO: check safety implications if hosting this on a server!
        self.ds = Dataset(df, time=ds.time, condition=ds.condition, features=self.features)
        self.sample_weights, self.sing_values, Vt = linalg.svd(np.asarray(self.ds.data[self.features]))
        self.feature_weights = Vt.T

        self.cluster = cluster
        if self.cluster is not None:
            self.labels = self.cluster.fit_predict(self.feature_weights[:, :cluster_components])

    def interpolate_sample_weights(self, component, interpolation='average', conditions=None, timepoints=None):
        """Compute interpolated sample weights for one component.
"""
        if conditions is None:
            conditions = self.condition_list
        else:
            for c in conditions:
                if c not in ds.condition_list:
                    raise ValueError(f'Condition "{c}" not found in dataset condition list: {self.condition_list}')
            self.condition_list = conditions
        weights = self.sample_weights[:, component]
        if timepoints is None:
            timepoints = self.ds.timepoints
        weights = np.zeros(len(timepoints))
        if interpolation == 'average':
            pass
        else:
            raise ValueError(f"""Only "interpolation='average'" is currently implemented, not "{interpolation}".""")
            
    def plot_singular_values(self, ax=None, trafo=None, fit_line=False, line_plot_args={}, **kwargs):
        """Plot the singular values.

Singular values are plotted in decreasing order in a new axes, or in one passed as argument to the function.
By default the plotstyle is '--.', but it can be changed with keyword arguments.
Optionally, a line can be fitted to (part of) the singular values, to indicate which ones are larger than one would expect from random data.

Parameters
---
ax : matplotlib Axes, default=None
    Axes to plot into, will be created if None.
trafo : callable, default=None
    Transformation to apply to the values before plotting, for example logarithm.
    The function needs to be vectorized (e.g., use numpy.log, not math.log).
fit_line : Boolean, float < 1, or int >= 1, default=False
    Fit a line to (part of) the (transformed) values and show it in the plot.
    With `fit_line=True`, a line is fitted to all values.
    If `fit_line` is an integer >= 1, fit the line to that number of smallest values.
    If it is a float < 1, fit the line to that fraction of smallest values.
line_plot_args : dictionary, default={}
    Dictionary of keyword arguments to pass to the plotting of the fitted line
**kwargs : 
    Arguments to be passed to the [`plot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html) method when plotting the singular values.

Returns
---
If a new figure is created (`ax=None`), then the resulting Figure and Axes object are returned.
"""
        if ax is None:
            fig, ax = pyplot.subplots(1, 1)
            if trafo is None:
                ax.set_ylabel("Singular values")
            else:
                ax.set_ylabel(str(trafo) + " singular values")
            ax.set_xlabel("Component index")
            new_fig = True
        else:
            new_fig = False
        if trafo is None:
            values = self.sing_values
        else:
            values = trafo(self.sing_values)
        ax.plot(values, '--.', **kwargs)
        if fit_line is not False:
            if fit_line is True:
                M = np.ones(shape=(self.sing_values.shape[0], 2))
                M[:,1] = np.arange(self.sing_values.shape[0])
                fit_vals = values
            elif fit_line < 1:
                num_fit = max(2, int(np.round(fit_line * self.sing_values.shape[0])))
                M = np.ones(shape=(num_fit, 2))
                M[:,1] = np.arange(self.sing_values.shape[0]-num_fit, self.sing_values.shape[0])
                fit_vals = values[-num_fit:]
            else:
                M = np.ones(shape=(fit_line, 2))
                M[:,1] = np.arange(self.sing_values.shape[0]-fit_line, self.sing_values.shape[0])
                fit_vals = values[-fit_line:]
            fit_param = linalg.lstsq(M, fit_vals)[0]
            M_full = np.ones(shape=(self.sing_values.shape[0], 2))
            M_full[:,1] = np.arange(self.sing_values.shape[0])
            ax.plot(M_full[:,1], M_full.dot(fit_param), **line_plot_args)
        if new_fig:
            return fig, ax

    def plot_features(self, ax=None, components=(0, 1), features=None, labels=False, plotstyle=None, **kwargs):
        """Component weight plot of the features.
"""
        if ax is None:
            fig, ax = pyplot.subplots(1, 1)
            ax.set_xlabel(f"Component {components[0]}")
            ax.set_ylabel(f"Component {components[1]}")
            new_fig = True
        else:
            new_fig = False
        if features is None:
            features = self.features
            feature_index = np.array(range(len(self.features)))
        else:
            feature_index = np.array([self.features.index(f) for f in features])
        if self.cluster is not None:
            for l in set(self.labels):
                ax.plot(self.feature_weights[feature_index[self.labels==l], components[0]], self.feature_weights[feature_index[self.labels==l], components[1]], '.')
        if labels is not False:
            if labels is True:
                for f, fi in zip(features, feature_index):
                    ax.annotate(f, ((self.feature_weights[fi, components[0]], self.feature_weights[fi, components[1]])))
        if new_fig:
            return fig, ax
                    

