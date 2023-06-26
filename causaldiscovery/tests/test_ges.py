import hashlib
import sys
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
sys.path.append("")
import unittest
import numpy as np

from vcausal.causaldiscovery.scorebased.ges import GESAlgorithm


######################################### Test Notes ###############################################
# The benchmark results for the loaded files (e.g., "./TestData/benchmark_returned_results/")      #
# were obtained from the causal-learn code at commit                                               #
# https://github.com/py-why/causal-learn/commit/5918419 (02-03-2022).                              #
# We cannot guarantee that the results are entirely "correct" and reflect the ground truth graph.  #
# Therefore, if your tests fail, it means that your modified code is logically inconsistent        #
# with the code as of commit 5918419, but it does not necessarily mean that your code is "wrong".  #
# If you are confident that your modification is "correct" (e.g., you fixed some bugs in 5918419), #
# please report it to us. We will then update these benchmark results accordingly. Thank you!      #
######################################### Test Notes ###############################################


BENCHMARK_TXTFILE_TO_MD5 = {
    "./TestData/data_linear_10.txt": "95a17e15038d4cade0845140b67c05a6",
    "./TestData/data_discrete_10.txt": "ccb51c6c1946d8524a8b29a49aef2cc4",
    "./TestData/graph.10.txt": "4970d4ecb8be999a82a665e5f5e0825b",
    "./TestData/test_ges_simulated_linear_gaussian_data.txt": "0d2490eeb9ee8ef3b18bf21d7e936e1e",
    "./TestData/test_ges_simulated_linear_gaussian_CPDAG.txt": "aa0146777186b07e56421ce46ed52914",
    "./TestData/benchmark_returned_results/linear_10_ges_local_score_BIC_none_none.txt": "3accb3673d2ccb4c110f3703d60fe702",
    "./TestData/benchmark_returned_results/discrete_10_ges_local_score_BDeu_none_none.txt": "eebd11747c1b927b2fdd048a55c8c3a5",
}

INCONSISTENT_RESULT_ERROR = "Returned graph is inconsistent with the benchmark. Please check your code with the commit 94d1536."
RESULT_PARITY_ERROR = "The returned graph does not align with the benchmark, suggesting an inconsistency in your code."

# verify files integrity first
for file_path, expected_MD5 in BENCHMARK_TXTFILE_TO_MD5.items():
    with open(file_path, 'rb') as fin:
        assert hashlib.md5(fin.read()).hexdigest() == expected_MD5,\
            f'{file_path} is corrupted. Please download it again from https://github.com/cmu-phil/causal-learn/blob/b51d788/tests/TestData'


class TestGES(unittest.TestCase):

    # GES with local_score_BIC on data from file "data_linear_10.txt".
    def test_GES_BIC_Linear(self):
        print('Initialising GES with BIC on Linear Data ...')
        data_path = "./TestData/data_linear_10.txt"
        truth_graph_path = "./TestData/graph.10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        truth_dag = txt2generalgraph(truth_graph_path)  # truth_dag is a GeneralGraph instance
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()

        # Using default parameters: score_func='local_score_BIC', maxP=None, parameters=None
        
        res_map = GESAlgorithm(data, score_func='local_score_BIC', maxP=None, parameters=None)  # res_map is Dict object，which contains the updated steps, the result causal graph and the result score.
        res_map.learn_GES()
        res_map.visualize_graph()
        benchmark_returned_graph = np.loadtxt(
            "./TestData/benchmark_returned_results/linear_10_ges_local_score_BIC_none_none.txt")
        assert np.all(res_map['G'].graph == benchmark_returned_graph), INCONSISTENT_RESULT_ERROR
        shd = SHD(truth_cpdag, res_map['G'])
        print(f"    ges(data, score_func='local_score_BIC', maxP=None, parameters=None)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        print('Test for GES with local_score_BIC on data passed!\n')

    # GES with default parameters on simulated linear Gaussian data.
    def test_GES_Default_Simulated_Linear(self):
        print('Initialising GES with default parameters on simulated linear Gaussian data ...')
        # Graph specification.
        num_of_nodes = 5
        truth_DAG_directed_edges = {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_directed_edges = {(0, 3), (1, 3), (2, 3), (2, 4), (3, 4)}
        truth_CPDAG_undirected_edges = {(0, 1), (1, 2), (2, 1), (1, 0)}
        truth_CPDAG = np.loadtxt("./TestData/test_ges_simulated_linear_gaussian_CPDAG.txt")

        ###### Simulation configuration: code to generate "./TestData/test_ges_simulated_linear_gaussian_data.txt" ######
        # np.random.seed(42)
        # linear_weight_minabs, linear_weight_maxabs, linear_weight_netative_prob = 0.5, 0.9, 0.5
        # sample_size = 10000
        # adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes))
        # adjacency_matrix[tuple(zip(*truth_DAG_directed_edges))] = 1
        # adjacency_matrix = adjacency_matrix.T
        # weight_mask = np.random.uniform(linear_weight_minabs, linear_weight_maxabs, (num_of_nodes, num_of_nodes))
        # weight_mask[np.unravel_index(np.random.choice(np.arange(weight_mask.size), replace=False,
        #                            size=int(weight_mask.size * linear_weight_netative_prob)), weight_mask.shape)] *= -1.
        # adjacency_matrix = adjacency_matrix * weight_mask
        # mixing_matrix = np.linalg.inv(np.eye(num_of_nodes) - adjacency_matrix)
        # exogenous_noise = np.random.normal(0, 1, (num_of_nodes, sample_size))
        # data = (mixing_matrix @ exogenous_noise).T
        # np.savetxt("./TestData/test_ges_simulated_linear_gaussian_data.txt", data)
        ###### Simulation configuration: code to generate "./TestData/test_ges_simulated_linear_gaussian_data.txt" ######

        data = np.loadtxt("./TestData/test_ges_simulated_linear_gaussian_data.txt")

        # Run GES with default parameters: score_func='local_score_BIC', maxP=None, parameters=None
        res_map = GESAlgorithm(data, score_func='local_score_BIC', maxP=None, parameters=None)
        res_map.learn_GES()
        res_map.visualize_graph()

        assert np.all(res_map['G'].graph == truth_CPDAG), RESULT_PARITY_ERROR
        print(f"    ges(data, score_func='local_score_BIC', maxP=None, parameters=None)\treturns exactly the same CPDAG as the truth.")

        print('Test for GES with default parameters on simulated linear Gaussian data passed!\n')

    # GES with local_score_BDeu on data from file "data_discrete_10.txt".
    def test_GES_BDeu_Discrete(self):
        print('Initialising GES with local_score_BDeu on discrete data ...')
        data_path = "./TestData/data_discrete_10.txt"
        truth_graph_path = "./TestData/graph.10.txt"
        data = np.loadtxt(data_path, skiprows=1)
        truth_dag = txt2generalgraph(truth_graph_path)  # truth_dag is a GeneralGraph instance
        truth_cpdag = dag2cpdag(truth_dag)
        num_edges_in_truth = truth_dag.get_num_edges()

        # Run GES with local_score_BDeu.
        res_map = GESAlgorithm(data, score_func='local_score_BDeu', maxP=None, parameters=None)
        res_map.learn_GES()
        res_map.visualize_graph()
        
        benchmark_returned_graph = np.loadtxt(
            "./TestData/benchmark_returned_results/discrete_10_ges_local_score_BDeu_none_none.txt")
        assert np.all(res_map['G'].graph == benchmark_returned_graph), INCONSISTENT_RESULT_ERROR
        shd = SHD(truth_cpdag, res_map['G'])
        print(f"    ges(data, score_func='local_score_BDeu', maxP=None, parameters=None)\tSHD: {shd.get_shd()} of {num_edges_in_truth}")

        print('Test for GES with local_score_BDeu on discrete data passed!\n')