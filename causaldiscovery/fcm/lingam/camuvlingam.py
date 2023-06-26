import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.FCMBased import lingam
from vcausal.casualdiscovery.utils.DAG2Adjacency_Matrix import DAG2Adjacency_Matrix

class CAMUVLiNGAMAlgorithm:
    def __init__(self, X):
        """
        Initialize the CAMUVLiNGAMAlgorithm class.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and n_features is the number of features.
        """
        self.X = X
        self.graph = None
        self.adjacency_matrix = None
        self.adjacency_matrix_generator = DAG2Adjacency_Matrix()

    def learn_CAMUVL(self, alpha, num_explanatory_vals):
        """
        Learn causal relationships using the CAMU-V LiNGAM algorithm.

        Parameters:
        - alpha: float
          The alpha level for independence testing.
        - num_explanatory_vals: int
          The maximum number of variables to infer causal relationships.

        Returns:
        - P: list of lists
          P[i] contains the indices of the parents of Xi.
        - U: list of tuples
          The indices of variable pairs having UCPs or UBPs.
        """
        P, U = lingam.CAMUV.execute(self.X, alpha, num_explanatory_vals)

        # Create a directed graph
        self.graph = nx.DiGraph()

        # Add nodes to the graph
        self.graph.add_nodes_from(range(len(P)))

        # Add edges to the graph based on the causal relationships
        for i, parents in enumerate(P):
            for parent in parents:
                self.graph.add_edge(parent, i)
        self.adjacency_matrix = self.adjacency_matrix_generator.get_adjacency_matrix(self.graph)
    
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
