import networkx as nx

class DAG2Adjacency_Matrix:
    def get_adjacency_matrix(self, graph):
        """
        Get the adjacency matrix of a graph.

        Parameters:
        - graph: networkx.DiGraph
          The directed graph.

        Returns:
        - adjacency_matrix: numpy.ndarray
          The adjacency matrix of the graph.
        """
        if graph is None:
            raise ValueError("DAG has not been learned yet.")

        adjacency_matrix = nx.to_numpy_array(graph, dtype=int)
        return adjacency_matrix
