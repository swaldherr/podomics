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

    def plot_features(self, ax=None, components=(0, 1), features=None, annotate=False, labels=None, **kwargs):
        """Component weight plot of the features.

Plots the weights of features in two components as a scatter plot.
Each data point in the plot corresponds to one feature, where the x-value is the weight of this feature in the first indicated component,
and the y-value is the weight of this feature in the second indicated component.

Parameters
---
ax : matplotlib Axes, default=None
    Axes to plot into, if `None` a new axis is created.
components : tuple of two int values, default=(0, 1)
    Components to consider for plotting.
    First element refer to the component weights to use for x-values, second element to the weights to use for y-values.
features : list of str, default=None
    List of features to include in the plot. If `None`, plot all features.
annotate : Boolean or list of feature identifiers, default=False
    Label the data points in the plot with the feature names, using the matplotlib `annotate` method.
    To label only selected features, pass the identifiers of the desired features as list.
labels : str, default=None
    Which labels to assign for a possible plot legend.
    By default, no labels are assigned.
    With `labels='clusters'`, labels according to the cluster numbers are assigned.
    If this argument is not `None` and a new figure is created, also shows the legend.
    If plotting in a given Axes, the legend has to be activated separately.
**kwargs : 
    Additional arguments are passed to matplotlib's `plot` function and can be used to change the plot style etc.
    If no arguments are given, the plotstyle `'.'` is used.

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata3.csv", sample="Sample", condition="Condition"))
    >>> fig, ax = pod_result.plot_features(labels='clusters', annotate=['0', '49'])
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
                fi = [i for i in feature_index if self.labels[i] == l]
                if len(fi) > 0:
                    x = self.feature_weights[fi, components[0]]
                    y = self.feature_weights[fi, components[1]]
                    label = f"Cluster {l}" if labels=='clusters' else None
                    if len(kwargs):
                        ax.plot(x, y, label=label, **kwargs)
                    else:
                        ax.plot(x, y, '.', label=label)
        else:
            if len(kwargs):
                ax.plot(self.feature_weights[feature_index, components[0]], self.feature_weights[feature_index, components[1]], **kwargs)
            else:
                ax.plot(self.feature_weights[feature_index, components[0]], self.feature_weights[feature_index, components[1]], '.')
        if annotate is not False:
            if annotate is True:
                annotate = features
            for f in annotate:
                fi = self.features.index(f)
                ax.annotate(f, ((self.feature_weights[fi, components[0]], self.feature_weights[fi, components[1]])))
        if new_fig:
            if labels is not None:
                ax.legend()
            return fig, ax
                    
    def plot_samples(self, ax=None, components=(0, 1), samples=None, annotate=False, labels=None, **kwargs):
        """Component weight plot of the samples.

Plots the weights of samples in two components as a scatter plot.
Each data point in the plot corresponds to one sample, where the x-value is the weight of this sample in the first indicated component,
and the y-value is the weight of this sample in the second indicated component.

Parameters
---
ax : matplotlib Axes, default=None
    Axes to plot into, if `None` a new axis is created.
components : tuple of two int values, default=(0, 1)
    Components to consider for plotting.
    First element refer to the component weights to use for x-values, second element to the weights to use for y-values.
samples : list of str, default=None
    List of samples (by IDs) to include in the plot. If `None`, plot all samples.
annotate : Boolean or list of feature identifiers, default=False
    Label the data points in the plot with the sample IDs, using the matplotlib `annotate` method.
    To label only selected samples, pass the IDs of the desired samples as list.
labels : str, default=None
    Can be used to either label conditions or timepoints for a plot legend.
    Options are:
    - `'condition'` : use condition names as labels.
    - `'time`' : use timepoint identifiers as labels.
    - `'both`' : include both condition and timepoint identifiers in the labels (only recommended with few samples / many replicates)
    When using that option, different conditions / timepoints are plotted with different colors, unless a custom plotstyle is used.
**kwargs : 
    Additional arguments are passed to matplotlib's `plot` function and can be used to change the plot style etc.
    If no arguments are given, the plotstyle `'.'` is used.

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata3.csv", sample="Sample", condition="Condition"))
    >>> fig, ax = pod_result.plot_samples(annotate=True, labels='condition')
"""
        if ax is None:
            fig, ax = pyplot.subplots(1, 1)
            ax.set_xlabel(f"Component {components[0]}")
            ax.set_ylabel(f"Component {components[1]}")
            new_fig = True
        else:
            new_fig = False
        if samples is None:
            samples = [s for s in self.ds.data.index]
            sample_index = np.array(range(len(samples)))
        else:
            sample_index = np.array([samples.index(s) for s in samples])
        data = self.sample_weights.iloc[sample_index, :]
        have_plot = False
        if labels is not None:
            if labels == 'condition':
                if self.ds.condition is not None:
                    for c in self.ds.condition_list:
                        plotsamples = data[data[self.ds.condition] == c]
                        if len(kwargs) > 0:
                            ax.plot(plotsamples[(components[0])], plotsamples[(components[1])], label=f"{self.ds.condition} {c}", **kwargs)
                        else:
                            ax.plot(plotsamples[(components[0])], plotsamples[(components[1])], '.', label=f"{self.ds.condition} {c}")
                    have_plot = True
            elif labels == 'time':
                for t in self.ds.timepoints:
                    plotsamples = data[data[self.ds.time] == t]
                    if len(kwargs) > 0:
                        ax.plot(plotsamples[(components[0])], plotsamples[(components[1])], label=f"{self.ds.time} {t}", **kwargs)
                    else:
                        ax.plot(plotsamples[(components[0])], plotsamples[(components[1])], '.', label=f"{self.ds.time} {t}")
                have_plot = True
            elif labels == 'both':
                if self.ds.condition is None:
                    self.plot_samples(ax=ax, components=components, samples=samples, labels='time', annotate=annotate, **kwargs)
                else:
                    for c in self.ds.condition_list:
                        for t in self.ds.timepoints:
                            plotsamples = data[(data[self.ds.condition] == c) & (data[self.ds.time] == t)]
                            if len(kwargs) > 0:
                                ax.plot(plotsamples[(components[0])], plotsamples[(components[1])], label=f"{c} @ {t}", **kwargs)
                            else:
                                ax.plot(plotsamples[(components[0])], plotsamples[(components[1])], '.', label=f"{c} @ {t}")
                have_plot = True
            else:
                raise ValueError(f"Option '{labels}' is not available for the 'labels' keyword argument.")
            if new_fig and have_plot:
                ax.legend()
        if not have_plot:
            if len(kwargs) > 0:
                ax.plot(data[(components[0])], data[(components[1])], **kwargs)
            else:
                ax.plot(data[(components[0])], data[(components[1])], '.')
        if annotate is not False:
            if annotate is True:
                annotate = samples
            for s in annotate:
                ax.annotate(s, ((data.loc[s, (components[0])], data.loc[s, (components[1])])))
        if new_fig:
            return fig, ax


    def plot_sample_weights(self, axs=None, components=[0], conditions=None, annotate=False, interpolate=False, interp_style={}, **kwargs):
        """Plot timecourse of sample weights.

A new figure is created by default.
To plot into an existing figure, pass a list of Axes in the `axs` argument.
Optionally an interpolation is added to the plot, using the `POD.interpolate_sample_weights()` method.
If multiple conditions are given, the plot elements are labelled with the condition identifier,
and if a new figure is created, a legend is shown.

Parameters
---
axs : list of matplotlib `Axes`, default=None
    Axes to plot into, must be one element for each component to be plotted.
components : list of int, default=[0]
    Indices of the components for which to plot time courses.
conditions : list of str, default=None
    Conditions to include in the plot (default: all)
interpolate : str, default=False
    Interpolation method to use, see `POD.interpolate_sample_weights()` for options.
annotate : Boolean or list of str, default=False
    Annotate data points with sample IDs. If `True`, annotate all samples, otherwise only the samples for which the IDs are given in the list.
interp_style : dict, default={}
    Arguments to pass to the `plot` method when plotting the interpolated time course. 
    Default is a simple line in the same color as the data points for this condition.
    Note that if you pass any arguments here, the color is not adjusted to the same as the data points!
**kwargs : additional arguments
    These are passed to the `plot` function when plotting the data points.
    Default is to use the `'.'` plotstyle.

Returns
---
Matplotlib figure and list of axes if these are newly created, otherwise nothing.

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata3.csv", sample="Sample", condition="Condition"))
    >>> fig, ax = pod_result.plot_sample_weights(annotate=True)

To plot a single condition in a previously created Axes:
    >>> from matplotlib import pyplot
    >>> fig, axs = pyplot.subplots(1, 2)
    >>> pod_result.plot_sample_weights(axs=axs, components=[0, 1], conditions=['a'], interpolate='average')
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
        if conditions is not None:
            weights = self.sample_weights[self.sample_weights[self.ds.condition].isin(conditions)]
        else:
            weights = self.sample_weights
        if interpolate is not False:
            method = interpolate if type(interpolate)==str else 'average'
            interp_weights = self.interpolate_sample_weights(interpolation=method, conditions=conditions)
        have_label = set()
        for comp, ax in zip(components, axs):
            if conditions is not None:
                for cond in conditions:
                    label = cond if cond not in have_label else None
                    if len(kwargs) > 0:
                        plot, = ax.plot(weights[weights[self.ds.condition]==cond][self.ds.time], weights[weights[self.ds.condition]==cond][comp], label=label, **kwargs)
                    else:
                        plot, = ax.plot(weights[weights[self.ds.condition]==cond][self.ds.time], weights[weights[self.ds.condition]==cond][comp], '.', label=label)
                    if interpolate is not False:
                        label = f"{cond} (interp.)" if cond not in have_label else None
                        cond_weights = interp_weights[interp_weights[self.ds.condition]==cond]
                        if len(interp_style) > 0:
                            ax.plot(cond_weights.index, cond_weights[comp], label=label, **interp_style)
                        else:
                            ax.plot(cond_weights.index, cond_weights[comp], label=label, color=plot.get_color())
                    have_label.add(cond)
            else:
                if len(kwargs) > 0:
                    plot, = ax.plot(weights[self.ds.time], weights[comp], **kwargs)
                else:
                    plot, = ax.plot(weights[self.ds.time], weights[comp], '.')
                if interpolate is not False:
                    if len(interp_style) > 0:
                        ax.plot(interp_weights.index, interp_weights[comp], **interp_style)
                    else:
                        ax.plot(interp_weights.index, interp_weights[comp], color=plot.get_color())
            if annotate is not False:
                if annotate is True:
                    annotate = weights.index
                for l in annotate:
                    i = weights.index.get_loc(l)
                    ax.annotate( l, (weights[self.ds.time][i], weights[comp][i]) )
        if new_fig:
            fig.legend()
            return fig, axs

        
    def plot_data_reconstruction(self, ax=None):
        """Plot data reconstruction
        """

    def plot_feature_trajectories(self, ax=None, components=(0, 1), conditions=None, features=None, interpolate=False, clusters=None, labels=False, **kwargs):
        """Plot trajectories of individual features in component space.

Arguments
---
conditions : list of str, default=None
    List of conditions to plot.

Example usage
---
    >>> pod_result = POD(dataset.read_csv("examples/exampledata1.csv", sample="Sample"))
    >>> fig, ax = pod_result.plot_feature_trajectories()
        """
        conditions = self._check_conditions(conditions)
        if ax is None:
            if conditions is not None:
                fig, axs = pyplot.subplots(1, len(conditions))
            else:
                fig, ax = pyplot.subplots(1, 1)
                axs = [ax,]
            for i, ax in enumerate(axs):
                ax.set_xlabel(f"Component #{components[0]}")
                if conditions is not None:
                    ax.set_title(f"{self.ds.condition} {conditions[i]}")
            axs[0].set_ylabel(f"Component #{components[1]}")
            new_fig = True
        else:
            axs = [ax,]
            new_fig = False

        if features is None:
            features = self.features
            feature_index = np.array(range(len(self.features)))
        else:
            feature_index = np.array([self.features.index(f) for f in features])

        if conditions is not None:
            weights = self.sample_weights[self.sample_weights[self.ds.condition].isin(conditions)]
        else:
            weights = self.sample_weights
        if interpolate is not False:
            method = interpolate if isinstance(interpolate, str) else 'average'
            interp_weights = self.interpolate_sample_weights(interpolation=method, conditions=conditions)

        for f, fi in zip(features, feature_index):
            if conditions is not None:
                for i,c in enumerate(conditions):
                    traj0 = self.sample_weights[components[0]] * self.feature_weights[fi, components[0]] * self.sing_values[components[0]]
                    traj1 = self.sample_weights[components[1]] * self.feature_weights[fi, components[1]] * self.sing_values[components[1]]
            else:
                traj0 = self.sample_weights[components[0]] * self.feature_weights[fi, components[0]] * self.sing_values[components[0]]
                traj1 = self.sample_weights[components[1]] * self.feature_weights[fi, components[1]] * self.sing_values[components[1]]
                axs[0].plot(traj0, traj1, '.', **kwargs)

        # if False: #self.cluster is not None:
        #     for l in set(self.labels):
        #         ax.plot(self.feature_weights[feature_index[self.labels==l], components[0]], self.feature_weights[feature_index[self.labels==l], components[1]], '.')
        # else:
        #     ax.plot(self.feature_weights[feature_index, components[0]], self.feature_weights[feature_index, components[1]], '.')
        if new_fig:
            return fig, ax
        
    def plot_feature_data_in(self, ax, components=(0, 1), features=None, condition=None, clusters=None, annotate=None, **kwargs):
        """Plot feature trajectories in given Axes.

Plot data from a single condition.
Plot only selected clusters.

Example usage
---
    >>> from matplotlib import pyplot
    >>> fig, ax = pyplot.subplots(1, 1)
    >>> pod_result = POD(dataset.read_csv("examples/exampledata1.csv", sample="Sample"))
    >>> pod_result.plot_feature_data_in(ax)
"""
        conditions = self._check_conditions(condition)
        condition = conditions[0] if conditions is not None else None
        df0 = self.get_component(components[0])
        df1 = self.get_component(components[1])
        if condition is not None and self.ds.condition is not None:
            df0 = df0[df0[self.ds.condition] == condition]
            df1 = df1[df1[self.ds.condition] == condition]
        if features is None:
            fs = self.features
        else:
            fs = features
        # features = np.asarray(features)
        if clusters is not None and self.cluster is not None:
            features = [f for f,l in zip(self.features, self.labels) if f in fs and l in clusters]
        else:
            features = fs
        labelled = False
        for f in features:
            if condition is not None and not labelled:
                ax.plot(df0[f], df1[f], '.', label=f"{self.ds.condition} {condition}", **kwargs)
                labelled = True
            else:
                ax.plot(df0[f], df1[f], '.', **kwargs)
            if annotate is not None:
                for a in annotate:
                    ax.annotate(f, (df0.loc[a,f], df1.loc[a,f]) )


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
        else:
            conditions = None
        return conditions
