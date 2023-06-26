import numpy as np
from causallearn.graph.Graph import Graph
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.SHD import SHD

class Evaluation:
    @staticmethod
    def evaluate_adjacency_confusion(truth: Graph, est: Graph):
        """
        Evaluate the adjacency confusion between two graphs.

        Parameters
        ----------
        truth : Graph
            Truth graph.
        est : Graph
            Estimated graph.

        Returns
        -------
        float
            Adjacency precision value.
        """
        adj_confusion = AdjacencyConfusion(truth, est)
        return adj_confusion.get_adj_precision()

    @staticmethod
    def evaluate_arrow_confusion(truth: Graph, est: Graph):
        """
        Evaluate the arrow precision between two graphs.

        Parameters
        ----------
        truth : Graph
            Truth graph.
        est : Graph
            Estimated graph.

        Returns
        -------
        float
            Arrow precision value.
        """
        arrow_confusion = ArrowConfusion(truth, est)
        return arrow_confusion.get_arrows_precision()

    @staticmethod
    def evaluate_shd(truth: Graph, est: Graph):
        """
        Evaluate the structural hamming distance (SHD) between two graphs.

        Parameters
        ----------
        truth : Graph
            Truth graph.
        est : Graph
            Estimated graph.

        Returns
        -------
        float
            Structural Hamming Distance (SHD) between the two graphs.
        """
        shd = SHD(truth, est)
        shd_score = float(shd.get_shd())
        return shd_score
    
    @staticmethod
    def evaluate_mse(true_adjacency_matrix, learned_adjacency_matrix):
        """
        Compute the mean squared error (MSE) between the true adjacency matrix and the learned adjacency matrix.

        Parameters:
        - true_adjacency_matrix: numpy.ndarray
        The true adjacency matrix.
        - learned_adjacency_matrix: numpy.ndarray
        The learned adjacency matrix.

        Returns:
        - mse_score: float
        The mean squared error between the true and learned adjacency matrices.
        """
        error = true_adjacency_matrix - learned_adjacency_matrix
        mse_score = np.mean(error ** 2)
        return mse_score
    