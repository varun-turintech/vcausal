import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import matplotlib.image as mpimg
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.GraphUtils import GraphUtils
from vcausal.casualdiscovery.utils.DAG2Adjacency_Matrix import DAG2Adjacency_Matrix

class GRaSPAlgorithm:
    def __init__(self, X, score_func='local_score_BIC', maxP=None, parameters=None):
        """
        Initialize the GRaSPAlgorithm class.

        Parameters:
        - X: numpy.ndarray, shape (n_samples, n_features)
            Data, where n_samples is the number of samples and n_features is the number of features.
        - score_func: str, optional (default='local_score_BIC')
            The score function you would like to use.
        - maxP: int, optional (default=None)
            Allowed maximum number of parents when searching the graph.
        - parameters: dict, optional (default=None)
            Additional parameters needed when using CV likelihood.
        """
        self.X = X
        self.score_func = score_func
        self.maxP = maxP
        self.parameters = parameters
        self.graph = None
        self.adjacency_matrix = None
        self.adjacency_matrix_generator = DAG2Adjacency_Matrix()

    def learn_GRaSP(self, depth=None):
        """
        Learn the GRaSP algorithm.

        Parameters:
        - depth: int, optional (default=None)
            The maximum depth of the search.

        Returns:
        - graph: networkx.DiGraph
            The learned causal DAG.
        """
        self.graph = grasp(self.X, self.score_func, depth, self.maxP, self.parameters)
        self.adjacency_matrix = self.adjacency_matrix_generator.get_adjacency_matrix(self.graph)
        return self.graph, self.adjacency_matrix

    def visualize_graph(self):
        """
        Visualize the learned DAG using pydot and display the image.
        """
        if self.graph is None:
            raise ValueError("DAG has not been learned yet.")

        pdy = GraphUtils.to_pydot(self.graph)
        tmp_png = pdy.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.axis('off')
        plt.imshow(img)
        plt.show()