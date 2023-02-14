"""Perform proper orthogonal decomposition for omics timeseries data."""

import numpy as np
from scipy import linalg
import pandas
import matplotlib
from matplotlib import pyplot

from . import dataset
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
sample_weights : DataFrame
    Pandas dataframe of sample weights in the POD.
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

Example usage
---
#### Load a dataset to run POD on

    >>> omics = dataset.read_csv("examples/exampledata3.csv", sample="Sample", condition="Condition")

#### Run POD on the dataset

To run POD on the full dataset, create a new `POD` object with the dataset as argument:

    >>> pod_result = POD(omics)

To use only a subset of the data, create the `POD` object with the appropriate lists in the `features` and/or `conditions` keyword arguments:

    >>> pod_result = POD(omics, conditions=['a'])

Clustering can be performed directly when running the analysis. Recommended clustering methods are [`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [`GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) from the [`sklearn`](https://scikit-learn.org) package.

    >>> from sklearn.cluster import KMeans
    >>> pod_result = POD(omics, cluster=KMeans(n_clusters=3, n_init='auto'), cluster_components=4)

    >>> from sklearn.mixture import GaussianMixture
    >>> pod_result = POD(omics, cluster=GaussianMixture(n_components=3), cluster_components=4)
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
            df = ds.data
        else:
            for c in conditions:
                if c not in ds.condition_list:
                    raise ValueError(f'Condition "{c}" not found in dataset condition list: {ds.condition_list}')
            self.condition_list = conditions
            df = ds.data[ds.data[ds.condition].isin(conditions)]

        self.ds = Dataset(df, time=ds.time, condition=ds.condition, features=self.features)
        U, self.sing_values, Vt = linalg.svd(np.asarray(self.ds.data[self.features]))
        self.feature_weights = Vt.T
        self.sample_weights = pandas.DataFrame(U, index=df.index)
        self.sample_weights.insert(0, self.ds.time, self.ds.data[self.ds.time])
        if self.ds.condition is not None:
            self.sample_weights.insert(0, self.ds.condition, self.ds.data[self.ds.condition])

        self.cluster = cluster
        if self.cluster is not None:
            self.labels = self.cluster.fit_predict(self.feature_weights[:, :cluster_components])

    def interpolate_sample_weights(self, interpolation='average', conditions=None, timepoints=None):
        """Compute interpolated sample weights.

Used to approximate a timecourse of component weights for the given conditions.
Depending on the interpolation method, for each condition, a time course of weights for each component is computed based on the weights of
individual samples from that condition.

Parameters
---
interpolation : str
    * 'average': average samples from the same time point, no interpolation among different time points
    Currently only this method is implemented.
conditions : list of str, default=None
    List of conditions to retain, default all.
timepoints : list of timepoint labels / array, default=None
    List of timepoints for which to compute interpolated weights, 
    default is the list of timepoints in the dataset.
    With `interpolation='average'`, the timepoints given here must also be contained in the dataset!

Returns
---
Dataframe with timepoints as index, and interpolated component weights as columns.
If a condition column is used, the result also contains that column information.
Columns labelled with integers from 0 correspond to interpolated weights for the components in order of decreasing singular values.

**Note:** If multiple conditions are in the dataset, the timepoint index in the result may not be unique!

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata1.csv", sample="Sample"))
    >>> average_weights = pod_result.interpolate_sample_weights()
    >>> average_weights.iloc[:2,:3]
                      0         1         2
    Timepoint                              
    0.00      -0.307343 -0.320584  0.245403
    0.25      -0.279958 -0.144988 -0.251014
"""
        conditions = self._check_conditions(conditions)
        if timepoints is None:
            timepoints = self.ds.timepoints
        if self.ds.condition is None:
            df = self.sample_weights
        else:
            df = self.sample_weights[self.sample_weights[self.ds.condition].isin(conditions)]
        if interpolation == 'average':
            df = df[df[self.ds.time].isin(timepoints)]
            if self.ds.condition is not None:
                res = [df[df[self.ds.condition]==c].groupby(self.ds.time).mean(numeric_only=True) for c in conditions]
                for i,c in enumerate(conditions):
                    res[i].insert(0, self.ds.condition, c)
                return pandas.concat(res)
            else:
                return df.groupby(self.ds.time).mean()
        else:
            raise ValueError(f"""Only "interpolation='average'" is currently implemented, not "{interpolation}".""")

    def get_component(self, component=0):
        """Compute reconstructed data for a single component.

This function computes the data reconstruction for a single component.
The result is returned in a

Parameters
---
component : int, default=0
    Index of the component for which to do the reconstruction.

Returns
---
DataFrame : 
    Pandas `DataFrame` with the same structure as the dataset that the POD was computed for.

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata1.csv", sample="Sample"))
    >>> df = pod_result.get_component(0)
    >>> df.iloc[:3, :4]
            Timepoint         0         1         2
    Sample                                         
    T0R0          0.0  0.803708  0.746042  0.753983
    T0R1          0.0  0.791919  0.735099  0.742923
    T0R2          0.0  0.811034  0.752842  0.760856
"""
        Ui = np.atleast_2d(self.sample_weights[component].to_numpy())
        Vi = np.atleast_2d(self.feature_weights[:,component])
        data = Ui.T.dot(Vi) * self.sing_values[component]
        df = pandas.DataFrame(data, columns=self.features, index=self.ds.data.index)
        df.insert(0, self.ds.time, self.ds.data[self.ds.time])
        if self.ds.condition is not None:
            df.insert(0, self.ds.condition, self.ds.data[self.ds.condition])
        return df


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

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata3.csv", sample="Sample", condition="Condition"))
    >>> fig, ax = pod_result.plot_singular_values(fit_line=0.7)
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

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata3.csv", sample="Sample", condition="Condition"))
    >>> fig, ax = pod_result.plot_features(labels=True)
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
        else:
            ax.plot(self.feature_weights[feature_index, components[0]], self.feature_weights[feature_index, components[1]], '.')
        if labels is not False:
            if labels is True:
                for f, fi in zip(features, feature_index):
                    ax.annotate(f, ((self.feature_weights[fi, components[0]], self.feature_weights[fi, components[1]])))
        if new_fig:
            return fig, ax
                    


    def plot_sample_weights(self, axs=None, components=[0], conditions=None, interpolate=False, labels=False, plotstyle={'marker':'.', 'linestyle':''}, interp_style={}, **kwargs):
        """Plot timecourse of sample weights.

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata3.csv", sample="Sample", condition="Condition"))
    >>> fig, ax = pod_result.plot_sample_weights(labels=True)
"""
        if axs is None:
            fig, axs = pyplot.subplots(1, len(components))
            new_fig = True
            if len(components) == 1:
                axs = [axs]
            for ax,c in zip(axs, components):
                ax.set_xlabel(self.ds.time)
                ax.set_title(f"Component #{c}")
            axs[0].set_ylabel("Weight")
        else:
            if len(components) == 1:
                axs = [axs]
            new_fig = False
        if len(components) != len(axs):
            raise ValueError(f"List `components` must have same length as `axs`, got {len(components)} vs {len(axs)}.")
        conditions = self._check_conditions(conditions)
        use_conditions = self.ds.condition is not None and conditions is not None
        if use_conditions:
            weights = self.sample_weights[self.sample_weights[self.ds.condition].isin(conditions)]
        else:
            weights = self.sample_weights
        if interpolate is not False:
            method = interpolate if type(interpolate)==str else 'average'
            interp_weights = self.interpolate_sample_weights(interpolation=method, conditions=conditions)
        for comp, ax in zip(components, axs):
            if use_conditions:
                for cond in conditions:
                    ax.plot(weights[weights[self.ds.condition]==cond][self.ds.time], weights[weights[self.ds.condition]==cond][comp], label=cond, **plotstyle)
                    if interpolate is not False:
                        cond_weights = interp_weights[interp_weights[self.ds.condition]==cond]
                        ax.plot(cond_weights.index, cond_weights[comp], label=f"{cond} (interp.)", **interp_style)
                ax.legend()
            else:
                ax.plot(weights[self.ds.time], weights[comp], **plotstyle)
                if interpolate is not False:
                    ax.plot(interp_weights.index, interp_weights[comp], **interp_style)
            if labels:
                for i, l in enumerate(weights.index):
                    ax.annotate( l, (weights[self.ds.time][i], weights[comp][i]) )
        if new_fig:
            return fig, axs
        


    def _check_conditions(self, conditions):
        """Check that all conditions are valid.

Returns
---
List of conditions to use.

Raises
---
`ValueError` if `conditions` is not a valid selection of conditions.
"""
        if self.ds.condition is not None:
            if conditions is None:
                conditions = self.condition_list
            else:
                for c in conditions:
                    if c not in self.ds.condition_list:
                        raise ValueError(f'Condition "{c}" not found in dataset condition list: {self.ds.condition_list}')
        return conditions
