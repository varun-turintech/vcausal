import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.FCMBased import lingam

class VarLiNGAMAlgorithm:
    def __init__(self, X):
        """
        Initialize the VarLiNGAMAlgorithm class.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        """
        self.X = X
        self.graphs = []

    def learn_VARL(
        self,
        lags=1,
        criterion='bic',
        prune=False,
        ar_coefs=None,
        lingam_model=None,
        random_state=None,
        top_k=3,
    ):
        """
        Learn the VARLiNGAM model.

        Parameters:
        - lags: int, optional (default=1)
            Number of lags.
        - criterion: str, optional (default='bic')
            Criterion to decide the best lags within 'lags'. Searching the best lags is disabled if 'criterion' is None.
        - prune: bool, optional (default=False)
            Whether to prune the adjacency matrix or not.
        - ar_coefs: array-like, optional (default=None)
            Coefficients of AR model. Estimating AR model is skipped if specified 'ar_coefs'. Shape must be ('lags', n_features, n_features).
        - lingam_model: lingam object, optional (default=None)
            LiNGAM model for causal discovery. If None, DirectLiNGAM algorithm is selected.
        - random_state: int, optional (default=None)
            'random_state' is the seed used by the random number generator.
        - top_k: int, optional (default=3)
            Number of top-k adjacency matrices to store.

        Returns:
        - adjacency_matrices: list of array-like, shape (n_features, n_features)
            The adjacency matrices B of the fitted model, where n_features is the number of features.
        - causal_order: array-like, shape (n_features)
            The causal order of the fitted model, where n_features is the number of features.
        - residuals: array-like, shape (n_samples)
            Residuals of regression, where n_samples is the number of samples.
        """
        VARL = lingam.VARLiNGAM(
            lags=lags,
            criterion=criterion,
            prune=prune,
            ar_coefs=ar_coefs,
            lingam_model=lingam_model,
            random_state=random_state,
        )
        VARL.fit(self.X)

        adjacency_matrices = VARL.adjacency_matrices_[:top_k]
        adjacency_matrix_ = VARL.adjacency_matrices[0]
        causal_order = VARL.causal_order_
        residuals = VARL.residuals_

        self.graphs = [nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph) for adj_matrix in adjacency_matrices]

        return adjacency_matrices, causal_order, residuals, adjacency_matrix_

    def visualize_graphs(self):
        """
        Visualize the learned causal DAGs using matplotlib and networkx.
        """
        if len(self.graphs) == 0:
            raise ValueError("No graphs have been learned yet.")

        for i, graph in enumerate(self.graphs):
            plt.figure(i + 1)
            pos = nx.spring_layout(graph)
            nx.draw(
                graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=500,
                arrowstyle="->",
                arrowsize=10,
                font_size=10,
            )
            plt.title("Causal DAG {}".format(i + 1))

        plt.show()