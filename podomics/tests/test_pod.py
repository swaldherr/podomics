"""
Tests for pod module
"""

from unittest import TestCase

import pandas
import os
import numpy as np
from sklearn import cluster
from scipy import linalg

from numpy import testing

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

class TestComputeComponent(TestCase):
    """Test computation of component values."""
    example_data = [f"examples/exampledata{i+1}.csv" for i in range(3)]
    examples = [pod.POD(dataset.read_csv(f, sample="Sample")) for f in example_data[:2]]
    examples.append( pod.POD(dataset.read_csv(example_data[2], sample="Sample", condition="Condition")) )

    def test_compute_component(self):
        """
        pod.POD.get_component(): testing component reconstruction.
        """
        for ex in self.examples:
            for i,s in enumerate(ex.sing_values):
                d_test = ex.get_component(component=i)
                U, S, Vt = linalg.svd(ex.ds.data[ex.features])
                d_ref = np.atleast_2d(U[:,i]).T.dot(np.atleast_2d(Vt[i,:])) * S[i]
                testing.assert_allclose(d_test[ex.features].to_numpy(), d_ref)

class TestFeaturePlot(TestCase):
    """Test plotting of features in component space."""
    example_data = [f"examples/exampledata{i+1}.csv" for i in range(3)]
    examples = [pod.POD(dataset.read_csv(f, sample="Sample"), cluster=cluster.KMeans(n_clusters=3, n_init='auto'), cluster_components=4) for f in example_data[:2]]
    examples.append( pod.POD(dataset.read_csv(example_data[2], sample="Sample", condition="Condition"), cluster=cluster.KMeans(n_clusters=3, n_init='auto'), cluster_components=4) )

    def test_default_call(self):
        """
        pod.POD.plot_features(): test default call.
        """
        for ex in self.examples:
            fig, ax = ex.plot_features()
            # testing with keyword argument
            fig, ax = ex.plot_features(marker='o')

    def test_with_labels(self):
        """
        pod.POD.plot_features(): test with labels.
        """
        for ex in self.examples:
            fig, ax = ex.plot_features(labels='clusters')

    def test_with_annotation(self):
        """
        pod.POD.plot_features(): test with labels.
        """
        for ex in self.examples:
            fig, ax = ex.plot_features(annotate=True)

    def test_feature_subset(self):
        """
        pod.POD.plot_features(): test plotting feature subset.        
        """
        for ex in self.examples:
            fig, ax = ex.plot_features(labels='clusters', features=['0', '49'])
        
            
class TestSamplePlot(TestCase):
    """Test plotting of samples in component space."""
    example_data = [f"examples/exampledata{i+1}.csv" for i in range(3)]
    examples = [pod.POD(dataset.read_csv(f, sample="Sample")) for f in example_data[:2]]
    examples.append( pod.POD(dataset.read_csv(example_data[2], sample="Sample", condition="Condition")) )

    def test_default_call(self):
        """
        pod.POD.plot_samples(): test default call.
        """
        for ex in self.examples:
            fig, ax = ex.plot_samples()
            # testing with keyword argument
            fig, ax = ex.plot_samples(marker='o')

    def test_with_labels(self):
        """
        pod.POD.plot_samples(): test with labels.
        """
        for ex in self.examples:
            fig, ax = ex.plot_samples(labels='condition')
            fig, ax = ex.plot_samples(labels='time')
            fig, ax = ex.plot_samples(labels='both')
            with self.assertRaises(ValueError):
                ex.plot_samples(labels='')

    def test_with_annotation(self):
        """
        pod.POD.plot_samples(): test with labels.
        """
        for ex in self.examples:
            fig, ax = ex.plot_samples(annotate=True)

            
class TestSampleWeightPlot(TestCase):
    """Test plotting of sample weights."""
    example_data = [f"examples/exampledata{i+1}.csv" for i in range(3)]
    examples = [pod.POD(dataset.read_csv(f, sample="Sample")) for f in example_data[:2]]
    examples.append( pod.POD(dataset.read_csv(example_data[2], sample="Sample", condition="Condition")) )

    def test_default_call(self):
        """
        pod.POD.plot_sample_weights(): test default call.
        """
        for ex in self.examples:
            fig, ax = ex.plot_sample_weights()

    def test_with_interpolated(self):
        """
        pod.POD.plot_sample_weights(): test with conditions.
        """
        for ex in self.examples:
            fig, ax = ex.plot_sample_weights(interpolate=True)
            
