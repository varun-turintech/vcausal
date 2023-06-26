import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from IPython.display import Image, display
from vcausal.casualdiscovery.utils.DAG2Adjacency_Matrix import DAG2Adjacency_Matrix

class FCIAlgorithm:
    def __init__(self, dataset):
        """
        Initialize the FCIAlgorithm class.

        Parameters:
        - dataset: numpy.ndarray, shape (n_samples, n_features)
            Data, where n_samples is the number of samples and n_features is the number of 		features.
        """
        self.dataset = dataset
        self.graph = None
        self.adjacency_matrix = None
        self.adjacency_matrix_generator = DAG2Adjacency_Matrix()

    def learn_FCI(
        self,
        independence_test_method='fisherz',
        alpha=0.05,
        depth=-1,
        max_path_length=-1,
        verbose=False,
        background_knowledge=None,
        cache_variables_map=None,
    ):
        """
        Learn the FCI algorithm.

        Parameters:
        - independence_test_method: str or function, optional (default='fisherz')
            Independence test method function.
        - alpha: float, optional (default=0.05)
            Significance level of individual partial correlation tests.
        - depth: int, optional (default=-1)
            The depth for the fast adjacency search, or -1 if unlimited.
        - max_path_length: int, optional (default=-1)
            The maximum length of any discriminating path, or -1 if unlimited.
        - verbose: bool, optional (default=False)
            Whether verbose output should be printed or logged.
        - background_knowledge: class BackgroundKnowledge, optional (default=None)
            Add prior edges according to assigned causal connections.
        - cache_variables_map: dict, optional (default=None)
            A map containing the variables related to cache.

        Returns:
        - graph: networkx.DiGraph
            The learned causal DAG.
        - edges: list
            Contains graph's edges properties.
        """
        G, edges = fci(
            self.dataset,
            independence_test_method,
            alpha,
            depth,
            max_path_length,
            verbose,
            background_knowledge,
            cache_variables_map,
        )
        self.graph = G
        self.adjacency_matrix = self.adjacency_matrix_generator.get_adjacency_matrix(self.graph)
        
        return G, edges, self.adjacency_matrix

    def visualize_graph(self, filename='causal_dag.png'):
        """
        Visualize the learned DAG using pydot and display the image.

        Parameters:
        - filename: str, optional (default='causal_dag.png')
            The filename for saving the image.
        """
        if self.graph is None:
            raise ValueError("DAG has not been learned yet.")

        pdy = GraphUtils.to_pydot(self.graph)
        pdy.write_png(filename)
        display(Image(filename=filename))