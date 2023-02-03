"""Dataset module for podomics."""

import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler

def read_csv(file, sample=None, time=None, condition=None, **kwargs):
    """
Import dataset from a CSV file.

TODO: info about file format.

Parameters
---
file : str, path, or buffer
    The csv file to read from.
sample : str, default=None
    Column header for sample labels. By default, no sample labels are used, and samples can only be indexed by integers.
time : str, default=None
    Column header for time stamps. 
    If not given, try to find a column header starting with "Time"; raise an error if not found.
condition : str, default=None
    Column header for conditions. By default, no conditions are considered.
**kwargs : 
    Are passed to [pandas.read_csv()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) and can be used to adapt to specifics in the file format, e.g.:
 * **`sep`**: separator string, pandas default is ','
 * **`comment`**: character that identifies lines to be interpreted as comments

Returns
---
Dataset : object
    The `Dataset` object constructed from the data in the given file.

Example usage:
---
    >>> omics = read_csv("examples/exampledata1.csv", sample='Sample')
    """
    df = pandas.read_csv(file, index_col=sample, **kwargs)
    return Dataset(df, time, condition)

class Dataset(object):
    """Data container and various data handling and plotting methods.
    """
    def __init__(self, data, time=None, condition=None, features=None):
        """The constructor creates a dataset from a pandas Dataframe.

Parameters
---
data : pandas DataFrame
    Contains the data to analyse.
    It should follow the column layout as explained under **`read_csv`**.
    If sample labels are used, they should be stored in the index of the DataFrame.
time : str, default=None
    Name of the column which contains the timestamps.
    If not given, try to find a column header starting with "Time"; raise an error if not found.
condition : str, default=None
    Name of the column which contains the conditions, no conditions are used if this is not given.
features : list of str, default=None
    Column headers to use as features. Use all remaining if not given.

Attributes
---
data : pandas DataFrame
    Access to the underlying DataFrame
time : str
    Column header used for timestamps
condition : str
    Column header used for condition labels
features : list of str
    Column headers that are referring to features in the dataset
condition_list : list of str
    List of unique condition labels found in the dataset
timepoints : array of numeric or str
    Sorted array of timestamps found in the dataset
scaling : object
    Class used for rescaling the data, see `Dataset.rescale`.

Example usage
---
    >>> df = pandas.read_csv("examples/exampledata1.csv", index_col="Sample")
    >>> omics = Dataset(df, time="Timepoint", features=['0', '1', '3', '4', '6'])
    >>> omics.time
    'Timepoint'
    >>> omics.timepoints
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        """
        self.data = data
        
        num_cols = len(self.data.columns)
        if time in self.data.columns:
            self.time = time
        else:
            if time is None:
                # try to infer time column
                for c in self.data.columns:
                    if c.startswith("Time"):
                        time = c
                if time is None:
                    raise ValueError(f"Could not identify a time column in {self.data.columns[:min(num_cols, 5)]}..., use the `time` parameter to specify one.")
                else:
                    self.time = time
            else:
                raise ValueError(f'Time column "{time}" not found in data columns: {self.data.columns[:min(num_cols, 5)]}...')

        if condition is None:
            self.condition_list = []
        else:
            if condition in self.data.columns:
                self.condition_list = list(set(self.data[condition]))
            else:
                raise ValueError(f'Condition column "{condition}" not found in data columns: {self.data.columns[:min(num_cols, 5)]}...')
        self.condition = condition
    
        if features is None:
            self.features = [f for f in self.data.columns if f not in (self.time, self.condition)]
        else:
            self.features = [f for f in features if (f in self.data.columns) and f not in (self.time, self.condition)]
            if len(self.features)==0:
                raise ValueError(f"Did not find any of the given feature names in data columns: {self.data.columns[:min(num_cols, 5)]}...")
        # Check that all feature columns are numeric
        # https://stackoverflow.com/questions/19900202/how-to-determine-whether-a-column-variable-is-numeric-or-not-in-pandas-numpy
        from pandas.api.types import is_numeric_dtype
        for f in self.features:
            if not is_numeric_dtype(self.data[f]):
                raise ValueError(f'Column "{f}" was interpreted as feature, but is not numeric:\n{self.data[f].head()}')

        # sort rows first by condition if used, then by timepoint
        if self.condition is not None:
            self.data.sort_values(by=[self.condition, self.time], inplace=True)
        else:
            self.data.sort_values(by=[self.time], inplace=True)

        self.scaling = None
        timepoints = self.data[self.time].unique()
        timepoints.sort()
        self.timepoints = timepoints


    def rescale(self, method=MaxAbsScaler(), conditions=None, features=None):
        """Rescale the dataset with a transformation method from [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing).

The data can optionally be filtered with the parameters **`conditions`** and **`features`** before the scaling.

The scaled result is returned as a new **`Dataset`** object.

Parameters
---
method : object
    Instance of a transformation class from [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) to be used for resaling the data.
    For median scaling, the class `MedianScaler` provided in this module can be used.

Example usage
---
Load a dataset:

    >>> omics = read_csv("examples/exampledata1.csv", sample="Sample")

#### Scaling with maximum absolute value
    >>> scaled = omics.rescale()
    >>> np.allclose([scaled.data[f].max() for f in scaled.features], 1.0)
    True

#### Scaling with median
    >>> scaled = omics.rescale(method=MedianScaler())
    >>> np.allclose([scaled.data[f].median() for f in scaled.features], 1.0)
    True
"""
        scaler = method
        if features is None:
            features = self.features
        if self.condition is None or conditions is None:
            df = self.data
        else:
            df = self.data.query(f"{self.condition} in @conditions")
        method.fit(df.loc[:, features])
        df_scaled = df.copy()
        df_scaled.loc[:, features] = method.transform(df.loc[:, features])
        data_scaled = Dataset(df_scaled, time=self.time, condition=self.condition, features=None)
        data_scaled.scaling = method
        return data_scaled

from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
    _check_sample_weight,
    FLOAT_DTYPES,
)
from sklearn.preprocessing._data import _handle_zeros_in_scale
class MedianScaler(MaxAbsScaler):
    """Scale each feature by its median value.

The implementation follows closely the one in [`sklearn.preprocessing.MaxAbsScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html).
However, sparse data matrices are not supported for fitting, only for the transformation.

The following documentation is replicated from `sklearn.preprocessing.MaxAbsScaler`. See there for more details and examples.

This estimator scales and translates each feature individually such
that the median value of each feature in the
training set will be 1.0. It does not shift/center the data, and
thus does not destroy any sparsity.

Parameters
----------
copy : bool, default=True
    Set to False to perform inplace scaling and avoid a copy (if the input
    is already a numpy array).
Attributes
----------
scale_ : ndarray of shape (n_features,)
    Per feature relative scaling of the data.
median_ : ndarray of shape (n_features,)
    Per feature median value.
n_features_in_ : int
    Number of features seen during `fit`.
feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during `fit`. Defined only when `X`
    has feature names that are all strings.
n_samples_seen_ : int
    The number of samples processed by the estimator.
"""
    def fit(self, X, y=None):
        """Compute the median value of X for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        self._reset()
        self._validate_params()

        X = self._validate_data(
            X,
            reset=True,
            accept_sparse=None,
            dtype=FLOAT_DTYPES,
            force_all_finite="allow-nan",
        )

        self.n_samples_seen_ = X.shape[0]
        self.median_ = np.nanmedian(X, axis=0)
        self.scale_ = _handle_zeros_in_scale(self.median_, copy=True)
        return self

    

    