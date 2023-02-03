"""
Tests for dataset module
"""

from unittest import TestCase

import pandas
import os
import numpy as np

from .. import dataset

class TestDatasetCreation(TestCase):
    """
    Test dataset creation
    """

    df_full = pandas.DataFrame({"Time":[0, 1], "Condition":["a", "a"], "F1":[1., 2.], "F2":[3., 4.]}, index=["S1", "S2"])
    def test_full(self):
        """
        dataset.Dataset(): test creation from full dataframe with different levels of info.
        """
        def checks(ds):
            self.assertTrue(type(ds.data) is pandas.DataFrame)
            self.assertTrue(ds.time == "Time")
            self.assertEqual(ds.features[0], "F1")
            self.assertEqual(ds.features[1], "F2")
        ds = dataset.Dataset(data=self.df_full, condition="Condition")
        checks(ds)
        ds = dataset.Dataset(data=self.df_full, time="Time", condition="Condition")
        checks(ds)
        ds = dataset.Dataset(data=self.df_full, condition="Condition")
        checks(ds)
        self.assertTrue(ds.condition == "Condition")
        self.assertEqual(ds.condition_list[0], "a")
        ds = dataset.Dataset(data=self.df_full, features=["F1", "F2"])
        checks(ds)

    df_fails = pandas.DataFrame({"T":[0, 1], "C":["a", "a"], "F1":[1., 2.], "F2":[3., 4.]})
    def test_except(self):
        """
        dataset.Dataset(): test exceptions
        """
        ds = dataset.Dataset(data=self.df_fails, time="T", condition="C")
        with self.assertRaises(ValueError):
            ds = dataset.Dataset(data=self.df_fails)
        with self.assertRaises(ValueError):
            ds = dataset.Dataset(data=self.df_fails, time="Ti")
        with self.assertRaises(ValueError):
            ds = dataset.Dataset(data=self.df_fails, time="T")
        with self.assertRaises(ValueError):
            ds = dataset.Dataset(data=self.df_fails, time="T", condition="Cond")
        with self.assertRaises(ValueError):
            ds = dataset.Dataset(data=self.df_fails, time="T", condition="C", features=["Fa", "Fb"])


class TestRescaling(TestCase):
    """
    Test the Dataset.rescale method.
    """
    df_conds = pandas.DataFrame({"Time":[0, 1, 0, 1], "Condition":["a", "a", "b", "b"], "F1":[1., 2., 3., 4.], "F2":[10., 20., 30., 40.]})
    def test_with_condition(self):
        """dataset.Dataset.rescale(): test with only some conditions
        """
        d = dataset.Dataset(data=self.df_conds, condition="Condition")
        ds = d.rescale()
        self.assertEqual(ds.data["F1"].max(), 1.0)
        self.assertEqual(ds.data["F2"].max(), 1.0)
        d.rescale(conditions=["a"])
        self.assertEqual(ds.data["F1"].max(), 1.0)
        self.assertEqual(ds.data["F2"].max(), 1.0)
        d.rescale(conditions=["a", "b"])
        self.assertEqual(ds.data["F1"].max(), 1.0)
        self.assertEqual(ds.data["F2"].max(), 1.0)

    def test_with_features(self):
        """dataset.Dataset.rescale(): test with only some features
        """
        d = dataset.Dataset(data=self.df_conds, condition="Condition")
        ds = d.rescale(features=["F1"])
        self.assertEqual(ds.data["F1"].max(), 1.0)
        self.assertFalse("F2" in ds.data.columns)
        d.rescale(conditions=["a"], features=["F1"])
        self.assertEqual(ds.data["F1"].max(), 1.0)

    def test_median_scaling(self):
        """dataset.Dataset.rescale(): test median scaler.
        """
        d = dataset.Dataset(data=self.df_conds, condition="Condition")
        ds = d.rescale(method=dataset.MedianScaler())
        for f in ds.features:
            self.assertEqual(ds.data[f].median(), 1.0)

    df_nans = pandas.DataFrame({"Time":[0, 1, 0, 1], "Condition":["a", "a", "b", "b"], "F1":[1., 2., 3., np.nan], "F2":[10., np.nan, 30., 40.]})
    def test_with_nans(self):
        """dataset.Dataset.rescale(): test with nan values
        """
        d = dataset.Dataset(data=self.df_nans, condition="Condition")
        ds = d.rescale(features=["F1"])
        self.assertEqual(ds.data["F1"].max(), 1.0)
        d.rescale(conditions=["a"], features=["F1"])
        self.assertEqual(ds.data["F1"].max(), 1.0)

class TestCsv(TestCase):
    """
    Test import of csv file.
    """
    def test_csv_import(self):
        """
        dataset.test_csv(): test csv import.
        """
        print("cwd:", os.getcwd())
        ds = dataset.read_csv("examples/exampledata1.csv", sample="Sample")
        self.assertTrue(type(ds.data) is pandas.DataFrame)
