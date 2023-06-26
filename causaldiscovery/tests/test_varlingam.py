import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd

from vcausal.causaldiscovery.fcm.lingam.varlingam import VarLiNGAMAlgorithm


class VarLiNGAMAlgorithmTestCase(unittest.TestCase):

    def test_VARLingam(self):
        X = pd.read_csv('tests/TestData/sample_data_var_lingam.csv')
        
        # Create an instance of the VarLiNGAMAlgorithmclass
        varl = VarLiNGAMAlgorithm(X)

        # Learn the causal DAG
        varl.learn_VARL()

        # Visualize the graph
        varl.visualize_graph()