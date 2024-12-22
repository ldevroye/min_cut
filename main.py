import random
from math import ceil, sqrt
from typing import Tuple, Any, List, Set

import networkx as nx
from dataclasses import dataclass


@dataclass
class Graph:
    edges: dict[int: Set[int]]  # edges[1] = [2,3,5,8] connected nodes
    nodes: List[int]

    def __init__(self):
        self.edges = dict()
        self.nodes = list()

    def num_edges(self):
        result: int = 0
        for k in self.edges.keys():
            result += len(self.edges[k])

        return result

    def add_node(self, k=1):
        n: int = 0
        if len(self.nodes) > 0:
            n = len(self.nodes)+1

        for i in range(n, n + k):
            self.nodes.append(i)
            self.edges[i] = set()

    def add_edge(self, u: int, v: int):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"edge exception {u} or {v} not in graph")

        if u not in self.edges[v] and \
                v not in self.edges[u]:
            self.edges[u].add(v)
            self.edges[v].add(u)

    def remove_edge(self, u, v):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"edge exception {u} or {v} not in graph")

        if u in self.edges[v] and \
            v in self.edges[u]:

            self.edges[u].remove(v)
            self.edges[v].remove(u)

    def contract(self, u, v):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Both nodes {u} and {v} must exist in the graph.")

        # Merge edges of v into u, avoiding self-loops
        self.edges[u].update(neighbor for neighbor in self.edges[v] if neighbor != u)
        if v in self.edges[u]:
            self.edges[u].remove(v)

        # Redirect all neighbors of v to point to u
        for neighbor in self.edges[v]:
            if neighbor != u:  # Skip self-loop
                self.edges[neighbor].remove(v)
                self.edges[neighbor].add(u)

        # Remove v from the graph
        del self.edges[v]
        self.nodes.remove(v)


@dataclass
class Solver:

    @staticmethod
    def contract(graph: nx.MultiGraph) -> nx.MultiGraph:
        """
        Karger's Contract algorithm for minimum cut using rustworkx.

        Returns:
            nx.MultiGraph: the cut of the two vertices left
        """
        local_graph = graph.copy()
        while len(local_graph.nodes) > 2:
            # Pick a random edge
            print("nodes : ", local_graph.nodes)
            print("edges: ", local_graph.edges)

            u, v = random.sample([local_graph.nodes], k=2)

            # Merge v into u
            nx.contracted_nodes(local_graph, u, v, self_loops=False)

        # the cut
        return local_graph

    @staticmethod
    def fast_cut(graph: nx.MultiGraph) -> nx.MultiGraph:
        """
        FastCut algorithm for minimum cut using rustworkx.

        Returns:
            nx.MultiGraph: THe smaller of the two cuts.
        """
        local_graph = graph.copy()

        def recursive_cut(subgraph: nx.MultiGraph):
            # 1.
            n = len(subgraph.nodes)

            # 2.
            if n <= 6:
                # Brute force: Run contract multiple times and take the best result
                return Solver.contract(subgraph)

            # a
            t = ceil(1 + n / sqrt(2))

            # b
            H1 = contract_until(subgraph, t)
            H2 = contract_until(subgraph, t)

            # c & d
            return min(recursive_cut(H1), recursive_cut(H2))

        def contract_until(graph: nx.MultiGraph, target_vertices: int) -> nx.MultiGraph:
            """Perform contractions until the graph has target_vertices left."""
            local_graph = graph.copy()
            while len(local_graph.nodes) > target_vertices:
                u, v = random.sample(local_graph.nodes, k=2)
                nx.contracted_nodes(local_graph, u, v, self_loops=False)

            return local_graph

        return recursive_cut(local_graph)

    @staticmethod
    def generate_random_graph(num_vertices, edge_probability=0.5):
        """Generates a random undirected graph using networkx."""
        graph = nx.MultiGraph()
        graph.add_nodes_from(range(num_vertices))
        num_edges = 0
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if random.random() < edge_probability:
                    graph.add_edge(i, j, num_edges)
                    num_edges += 1
        return graph

    @staticmethod
    def generate_random_graph_2(num_vertices, edge_probability=0.5) -> Graph:
        """Generates a random undirected graph using networkx."""
        result: Graph = Graph()
        result.add_node(num_vertices)
        num_edges = 0
        for i in range(0, num_vertices):
            for j in range(i + 1, num_vertices):
                if random.random() < edge_probability:
                    result.add_edge(i, j)
                    num_edges += 1
        return result

# Example usage
if __name__ == "__main__":
    random.seed(523920)

    # Generate a random graph
    graph = Solver.generate_random_graph_2(10, 0.7)

    print("BEFORE Graph edges: ", graph.edges, graph.num_edges(), "\nGraph nodes :", graph.nodes, len(graph.nodes))

    graph.contract(1, 2)

    print("AFTER Graph edges: ", graph.edges, graph.num_edges(), "\nGraph nodes :", graph.nodes, len(graph.nodes))

    """    
    # Run Karger's algorithm
    karger_cut = Solver.contract(graph)
    print("Karger's Algorithm Cut:", karger_cut)

    print("Graph edges 2: ", graph.edges)

    # Run FastCut algorithm
    fast_cut_result = Solver.fast_cut(graph)
    print("FastCut Algorithm Cut:", fast_cut_result)
    """