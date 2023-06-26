import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.FCMBased import lingam

class RCDLiNGAMAlgorithm:
    def __init__(self, X):
        """
        Initialize the RCDLiNGAMAlgorithm class.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        """
        self.X = X
        self.graph = None
        self.adjacency_matrix = None

    def learn_RCD(
        self,
        max_explanatory_num=2,
        cor_alpha=0.01,
        ind_alpha=0.01,
        shapiro_alpha=0.01,
        MLHSICR=False,
        bw_method='mdbs',
    ):
        """
        Learn the RCDLiNGAM model.

        Parameters:
        - max_explanatory_num: int, optional (default=2)
            Maximum number of explanatory variables.
        - cor_alpha: float, optional (default=0.01)
            Alpha level for Pearson correlation.
        - ind_alpha: float, optional (default=0.01)
            Alpha level for HSIC.
        - shapiro_alpha: float, optional (default=0.01)
            Alpha level for Shapiro-Wilk test.
        - MLHSICR: bool, optional (default=False)
            If True, use MLHSICR for multiple regression; if False, use OLS for multiple regression.
        - bw_method: str, optional (default='mdbs')
            The method used to calculate the bandwidth of the HSIC.

        Returns:
        - adjacency_matrix: array-like, shape (n_features, n_features)
            The adjacency matrix B of the fitted model, where n_features is the number of features.
        - ancestors_list: array-like, shape (n_features)
            The list of causal ancestors sets, where n_features is the number of features.
        """
        rcd = lingam.RCD(
            max_explanatory_num=max_explanatory_num,
            cor_alpha=cor_alpha,
            ind_alpha=ind_alpha,
            shapiro_alpha=shapiro_alpha,
            MLHSICR=MLHSICR,
            bw_method=bw_method,
        )
        rcd.fit(self.X)
        self.graph = nx.DiGraph(rcd.adjacency_matrix_)
        self.adjacency_matrix = rcd.adjacency_matrix_
    
    def visualize_graph(self):
        """
        Visualize the learned causal DAG using matplotlib and networkx.
        """
        if self.graph is None:
            raise ValueError("Graph has not been learned yet.")

        pos = nx.circular_layout(self.graph)
        nx.draw_circular(self.graph, with_labels=True, pos=pos)
        plt.show()