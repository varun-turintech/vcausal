import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd

from vcausal.causaldiscovery.fcm.lingam.rcd import RCDLiNGAMAlgorithm


class RCDLiNGAMAlgorithmTestCase(unittest.TestCase):

    def test_DL(self):
        np.set_printoptions(precision=3, suppress=True)
        np.random.seed(100)
        n_samples = 1000
        x0 = np.random.uniform(size=n_samples)
        x3 = 2.0 * x0 + np.random.uniform(size=n_samples)
        x2 = 0.5 * x3 + np.random.uniform(size=n_samples)
        x1 = -3.0 * x0 + 1.5 * x2 + np.random.uniform(size=n_samples)
        x5 = -2.5 * x1 + 0.8 * x2 + np.random.uniform(size=n_samples)
        x4 = -2.0 * x0 - 1.0 * x1 + 4.0 * x3 + np.random.uniform(size=n_samples)
        X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])

         # Create an instance of the RCDLiNGAMAlgorithm class
        rcd = RCDLiNGAMAlgorithm(X)

        # Learn the causal DAG
        rcd.learn_RCD()

        # Visualize the graph
        rcd.visualize_graph()