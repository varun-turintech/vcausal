import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from vcausal.casualdiscovery.utils.DAG2Adjacency_Matrix import DAG2Adjacency_Matrix

class PCAlgorithm:
    def __init__(self, data):
        """
        Initialize the PCAlgorithm class.

        Parameters:
        - data: numpy.ndarray, shape (n_samples, n_features)
            Data, where n_samples is the number of samples and n_features is the number of features.
        """
        self.data = data
        self.graph = None
        self.adjacency_matrix = None
        self.adjacency_matrix_generator = DAG2Adjacency_Matrix()

    def learn_PC(
        self,
        alpha=0.05,
        indep_test='fisherz',
        stable=True,
        uc_rule=0,
        uc_priority=2,
        mvpc=False,
        correction_name='MV_Crtn_Fisher_Z',
        background_knowledge=None,
        verbose=False,
        show_progress=True
    ):
        """
        Learn the PC algorithm.

        Parameters:
        - alpha: float, optional (default=0.05)
            Desired significance level in (0, 1).
        - indep_test: str, optional (default='fisherz')
            Name of the independence test method.
        - stable: bool, optional (default=True)
            Whether to run stabilized skeleton discovery or not.
        - uc_rule: int, optional (default=0)
            How unshielded colliders are oriented.
        - uc_priority: int, optional (default=2)
            Rule of resolving conflicts between unshielded colliders.
        - mvpc: bool, optional (default=False)
            Whether to use missing-value PC or not.
        - correction_name: str, optional (default='MV_Crtn_Fisher_Z')
            Missing value correction if using missing-value PC.
        - background_knowledge: class BackgroundKnowledge, optional (default=None)
            Add prior edges according to assigned causal connections.
        - verbose: bool, optional (default=False)
            Whether to print verbose output.
        - show_progress: bool, optional (default=True)
            Whether to show the algorithm progress in the console.

        Returns:
        - graph: networkx.DiGraph
            The learned causal DAG.
        """
        self.graph = pc(
            self.data,
            alpha=alpha,
            indep_test=indep_test,
            stable=stable,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            mvpc=mvpc,
            correction_name=correction_name,
            background_knowledge=background_knowledge,
            verbose=verbose,
            show_progress=show_progress
        )
        self.adjacency_matrix = self.adjacency_matrix_generator.get_adjacency_matrix(self.graph)
        return self.graph, self.adjacency_matrix

    def visualize_dag(self):
        """
        Visualize the learned DAG using matplotlib and networkx.
        """
        if self.graph is None:
            raise ValueError("DAG has not been learned yet.")

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