import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.FCMBased import lingam

class DirectLiNGAMAlgorithm:
    def __init__(self, X):
        """
        Initialize the DirectLiNGAMAlgorithm class.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        """
        self.X = X
        self.graph = None

    def learn_DL(
        self, random_state=None, prior_knowledge=None, apply_prior_knowledge_softly=False, measure="pwling"
    ):
        """
        Learn the DirectLiNGAM model.

        Parameters:
        - random_state: int, optional (default=None)
            The seed used by the random number generator.
        - prior_knowledge: array-like, shape (n_features, n_features), optional (default=None)
            Prior knowledge used for causal discovery, where n_features is the number of features.
        - apply_prior_knowledge_softly: boolean, optional (default=False)
            If True, apply prior knowledge softly.
        - measure: {'pwling', 'kernel'}, optional (default='pwling')
            Measure to evaluate independence: 'pwling' or 'kernel'.
        """
        DL = lingam.DirectLiNGAM(
            random_state=random_state,
            prior_knowledge=prior_knowledge,
            apply_prior_knowledge_softly=apply_prior_knowledge_softly,
            measure=measure
        )
        DL.fit(self.X)
        self.graph = nx.from_numpy_array(DL.adjacency_matrix_, create_using=nx.DiGraph)
        self.adjacency_matrix = DL.adjacency_matrix_
    
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
            node_color="lightblue",
            node_size=500,
            arrowstyle="->",
            arrowsize=10,
            font_size=10,
        )
        plt.title("Causal DAG")
        plt.show()

