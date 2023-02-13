"""
Tests for pod module
"""

from unittest import TestCase

import pandas
import os
import numpy as np
from sklearn import cluster

from .. import pod
from .. import dataset

class TestRunPOD(TestCase):
    """
    Test constructor for POD class
    """
    example_data = [f"examples/exampledata{i+1}.csv" for i in range(3)]
    example_ds = [dataset.read_csv(f, sample="Sample") for f in example_data[:2]]
    example_ds.append( dataset.read_csv(example_data[2], sample="Sample", condition="Condition") )

    def test_default_call(self):
        """
        pod.POD(): test default constructor.
        """
        for ds in self.example_ds:
            pod.POD(ds)

    def test_selected_features(self):
        """
        pod.POD(): test constructor with selected features.
        """
        features = [f"{i}" for i in range(20)]
        for ds in self.example_ds:
            p = pod.POD(ds, features=features)
            self.assertTrue("0" in p.ds.features)
            self.assertFalse("30" in p.ds.features)

    def test_wrong_feature(self):
        """
        pod.POD(): test constructor with features not in data.
        """
        for ds in self.example_ds:
            with self.assertRaises(ValueError):
                pod.POD(ds, features=["NOT", "IN", "DATA"])

    def test_selected_condition(self):
        """
        pod.POD(): test constructor with selected condition.
        """
        p = pod.POD(self.example_ds[2], conditions=["a"])
        self.assertTrue("a" in p.ds.condition_list)
        self.assertFalse("b" in p.ds.condition_list)

    def test_wrong_condition(self):
        """
        pod.POD(): test constructor with condition not in data.
        """
        with self.assertRaises(ValueError):
            pod.POD(self.example_ds[2], conditions=["c"])

    def test_with_clustering(self):
        """
        pod.POD(): test constructor with clustering (KMeans from sklearn).
        """
        for ds in self.example_ds:
            p = pod.POD(ds, cluster=cluster.KMeans(n_clusters=3, n_init='auto'))
            p = pod.POD(ds, cluster=cluster.KMeans(n_clusters=3, n_init='auto'), cluster_components=2)


class TestInterpolateSamples(TestCase):
    """
    Test interpolation of sample weights.
    """
    example_data = [f"examples/exampledata{i+1}.csv" for i in range(3)]
    examples = [pod.POD(dataset.read_csv(f, sample="Sample")) for f in example_data[:2]]
    examples.append( pod.POD(dataset.read_csv(example_data[2], sample="Sample", condition="Condition")) )

    def test_default_call(self):
        """
        pod.POD.interpolate_sample_weights(): test default call.
        """
        for ex in self.examples:
            w = ex.interpolate_sample_weights()
            self.assertEqual(len(set(w.index)), len(ex.ds.timepoints))

    def test_selected_condition(self):
        """
        pod.POD.interpolate_sample_weights(): test with only selected condition
        """
        w = self.examples[2].interpolate_sample_weights(conditions=['a'])
        self.assertEqual(len(w.index), len(self.examples[2].ds.timepoints))
        for wi, di in zip(w.index, self.examples[2].ds.timepoints):
            self.assertEqual(wi,  di)

    def test_wrong_condition(self):
        """
        pod.POD.interpolate_sample_weights(): test with condition not in dataset
        """
        with self.assertRaises(ValueError):
            self.examples[2].interpolate_sample_weights(conditions=["c"])

    def test_selected_timepoints(self):
        """
        pod.POD.interpolate_sample_weights(): test with subset of timepoints
        """
        for ex in self.examples:
            timepoints = ex.ds.timepoints[:3]
            w = ex.interpolate_sample_weights(timepoints=timepoints)
            self.assertEqual(len(set(w.index)), len(timepoints))
            if len(w.index) == len(timepoints):
                for wi, ti in zip(w.index, timepoints):
                    self.assertEqual(wi, ti)
