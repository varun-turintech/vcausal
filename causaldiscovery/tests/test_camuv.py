import sys

sys.path.append("")
import random
import unittest

import numpy as np

from vcausal.causaldiscovery.fcm.lingam.camuvlingam import CAMUVLiNGAMAlgorithm


def get_random_constant(v, r):
    constant = random.uniform(-1.0, 1.0)
    if constant > 0:
        constant = random.uniform(v, r)
    else:
        constant = random.uniform(-r, -v)
    return constant


def get_noise(n):
    noise = ((np.random.rand(1, n) - 0.5) * 5).reshape(n)
    mean = get_random_constant(0.0, 2.0)
    noise += mean
    return noise


def causal_func(cause):
    x = get_random_constant(-5.0, 5.0)
    y = get_random_constant(-1.0, 1.0)
    z = int(random.uniform(2, 3))
    return ((cause + x) ** (z)) + y


def create_data(n):
    causal_pairs = [[0, 1], [0, 3], [2, 4], [5,7]]
    intermediate_pairs = [[2, 5], [3,6]]
    confounder_pairs = [[3, 4]]

    n_variables = 8

    data = np.zeros((n, n_variables))  # observed data
    confounders = np.zeros((n, len(confounder_pairs)))  # data of unobserced common causes

    # Addition of external effects
    for i in range(n_variables):
        data[:, i] = get_noise(n)
    for i in range(len(confounder_pairs)):
        confounders[:, i] = get_noise(n)
        confounders[:, i] = confounders[:, i] / np.std(confounders[:, i])

    # Addition of the effects of unobserved common causes
    for i, cpair in enumerate(confounder_pairs):
        cpair = list(cpair)
        cpair.sort()
        data[:, cpair[0]] += causal_func(confounders[:, i])
        data[:, cpair[1]] += causal_func(confounders[:, i])

    for i1 in range(n_variables)[0:n_variables]:
        data[:, i1] = data[:, i1] / np.std(data[:, i1])
        for i2 in range(n_variables)[i1 + 1:n_variables + 1]:
            # Adding direct effects between observed variables
            if [i1, i2] in causal_pairs:
                data[:, i2] += causal_func(data[:, i1])
            # Adding undirected effects between observed variables mediated through unobserved variables
            if [i1, i2] in intermediate_pairs:
                interm = causal_func(data[:, i1]) + get_noise(n)
                interm = interm / np.std(interm)
                data[:, i2] += causal_func(interm)

    return data


class CAMUVAlgorithmTestCase(unittest.TestCase):

    def test_learn_CAMUVL(self):
        sample_size = 3000
        data = create_data(sample_size)
        
        # Create an instance of the CAMUVLiNGAMAlgorithm class
        camuv = CAMUVLiNGAMAlgorithm(data)
        
        # Learn causal relationships using the CAMU-V LiNGAM algorithm
        alpha = 0.01
        num_explanatory_vals = 3

        P, U = camuv.learn_CAMUVL(alpha,num_explanatory_vals)

        for i, result in enumerate(P):
            if not len(result) == 0:
                print("child: " + str(i) + ",  parents: " + str(result))

        for result in U:
            print(result)