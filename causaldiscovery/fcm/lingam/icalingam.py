import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.FCMBased import lingam

class IcaLiNGAMAlgorithm:
    def __init__(self, X):
        """
        Initialize the IcaLiNGAMAlgorithm class.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        """
        self.X = X
        self.graph = None

    def learn_ICAL(self, random_state=None, max_iter=1000):
        """
        Learn the ICALiNGAM model.

        Parameters:
        - random_state: int, optional (default=None)
            The seed used by the random number generator.
        - max_iter: int, optional (default=1000)
            The maximum number of iterations of FastICA.
        """
        ICAL = lingam.ICALiNGAM(random_state=random_state, max_iter=max_iter)
        ICAL.fit(self.X)
        self.graph = nx.from_numpy_array(ICAL.adjacency_matrix_, create_using=nx.DiGraph)
        self.adjacency_matrix = ICAL.adjacency_matrix_

    def visualize_graph(self):
        """
        Visualize the learned causal DAG using matplotlib and networkx.
        """
        if self.graph is None:
            raise ValueError("Graph has not been learned yet.")
        
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color='lightblue',
            node_size=500,
            arrowstyle='->',
            arrowsize=10,
            font_size=10
        )
        plt.title('Causal DAG')
        plt.show()

