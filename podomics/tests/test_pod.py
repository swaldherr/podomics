"""
Tests for dataset module
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

    def test_selected_condition(self):
        """
        pod.POD(): test constructor with selected condition.
        """
        p = pod.POD(self.example_ds[2], conditions=["a"])
        self.assertTrue("a" in p.ds.condition_list)
        self.assertFalse("b" in p.ds.condition_list)

    def test_with_clustering(self):
        """
        pod.POD(): test constructor with clustering (KMeans from sklearn).
        """
        for ds in self.example_ds:
            p = pod.POD(ds, cluster=cluster.KMeans(n_clusters=3, n_init='auto'))
            p = pod.POD(ds, cluster=cluster.KMeans(n_clusters=3, n_init='auto'), cluster_components=2)

        
