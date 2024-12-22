import random
from math import ceil, sqrt
import rustworkx as rx
from dataclasses import dataclass

@dataclass
class Resolve:
    graph: rx.PyGraph

    def contract(self) -> int:
        """
        Karger's Contract algorithm for minimum cut using rustworkx.

        Returns:
            int: The size of the cut.
        """
        # Make a deep copy of the graph to avoid modifying the original
        local_graph: rx.PyGraph = self.graph.copy()

        while local_graph.num_nodes() > 2:
            # Pick a random edge
            u, v = random.choice(local_graph.edge_list())

            # Merge v into u
            local_graph.contract_nodes(u, v, merge_nodes=False)

        # Remaining edges determine the cut size
        return len(local_graph.edge_list())

    def fast_cut(self):
        """
        FastCut algorithm for minimum cut using rustworkx.

        Returns:
            int: The size of the minimum cut.
        """
        def recursive_cut(subgraph):
            n = subgraph.num_nodes()
            if n <= 6:
                # Brute force: Run contract multiple times and take the best result
                return min(Resolve(subgraph).contract() for _ in range(n**2))

            t = ceil(1 + n / sqrt(2))
            contracted1 = contract_until(subgraph, t)
            contracted2 = contract_until(subgraph, t)

            return min(recursive_cut(contracted1), recursive_cut(contracted2))

        def contract_until(graph, target_vertices):
            """Perform contractions until the graph has target_vertices left."""
            local_graph = graph.copy()
            while local_graph.num_nodes() > target_vertices:
                u, v = random.choice(local_graph.edge_list())
                local_graph.contract_nodes(u, v, merge_nodes=False)

            return local_graph

        return recursive_cut(self.graph)

    @staticmethod
    def generate_random_graph(num_vertices, edge_probability=0.5):
        """Generates a random undirected graph using rustworkx."""
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_vertices))
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if random.random() < edge_probability:
                    graph.add_edge(i, j, None)
        return graph


# Example usage
if __name__ == "__main__":
    random.seed(42)

    # Generate a random graph
    graph = Resolve.generate_random_graph(10, 0.5)

    solver = Resolve(graph)

    print("Graph edges:", graph.edge_list())

    # Run Karger's algorithm
    karger_cut = solver.contract()
    print("Karger's Algorithm Cut:", karger_cut)

    # Run FastCut algorithm
    fast_cut_result = solver.fast_cut()
    print("FastCut Algorithm Cut:", fast_cut_result)
