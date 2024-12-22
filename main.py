import random
from math import ceil, sqrt
from typing import Tuple, Any, List, Set

from copy import deepcopy

import networkx as nx
from dataclasses import dataclass


@dataclass
class Graph:
    edges: dict[int: List[int]]  # edges[1] = [2,3,5,8] connected nodes
    nodes: List[int]

    def __init__(self, edges: dict, nodes: list):
        self.edges: dict = edges
        self.nodes: list = nodes

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Graph({node: neighbors.copy() for node, neighbors in self.edges.items()},
                     self.nodes.copy())

    def __str__(self):
        str_edges = "{\n"
        for k in self.edges.keys():
            str_edges += f"\t{k}: {self.edges[k]}\n"
        str_edges += "\t}"

        return f"\nEdges: {str_edges}, {self.num_edges()} \nNodes : {self.nodes}, {self.num_nodes()}"

    def num_edges(self) -> int:
        result: int = 0
        for k in self.edges.keys():
            result += len(self.edges[k])

        return int(result / 2)

    def num_nodes(self) -> int:
        return len(self.nodes)

    def add_node(self, k=1):
        n: int = 0
        if len(self.nodes) > 0:
            n = len(self.nodes) + 1

        for i in range(n, n + k):
            self.nodes.append(i)
            self.edges[i] = []

    def add_edge(self, u: int, v: int):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"edge exception {u} or {v} not in graph")

        if u not in self.edges[v] and \
                v not in self.edges[u]:
            self.edges[u].append(v)
            self.edges[v].append(u)

    def remove_edge(self, u, v):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"edge exception {u} or {v} not in graph")

        if u in self.edges[v] and \
                v in self.edges[u]:
            self.edges[u].remove(v)
            self.edges[v].remove(u)

    def get_random_edge(self) -> Tuple[int, int]:
        u = random.choice(self.nodes)

        while len(self.edges[u]) < 1:
            u = random.choice(self.nodes)

        v = random.choice(list(self.edges[u]))
        return u, v

    def contract(self, u, v):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Both nodes {u} and {v} must exist in the graph.")
        starting_num = self.num_edges()

        # Merge edges of v into u, avoiding self-loops
        self.edges[u].extend(neighbor for neighbor in self.edges[v] if neighbor != u)

        # Redirect all neighbors of v to point to u
        for neighbor in self.edges[v]:
            if neighbor != u:  # Skip self-loop
                self.edges[neighbor] = [u if x == v else x for x in self.edges[neighbor]]
            else:
                self.edges[u].remove(v)

        # Remove v from the graph
        del self.edges[v]
        self.nodes.remove(v)

        if starting_num-self.num_edges() != 1:
            raise ValueError(f"wrong number of contraction, != 1 edge diff {starting_num} before and {self.num_edges()} after")

    def contract_random(self):
        u, v = self.get_random_edge()
        print(u, v)
        self.contract(u, v)


class Solver:

    @staticmethod
    def contract(graph: Graph) -> Graph:
        """
        Karger's Contract algorithm for minimum cut using rustworkx.

        Returns:
            Graph: the cut of the two vertices left
        """
        local_graph = graph.copy()
        while len(local_graph.nodes) > 2:
            # Merge v into u
            local_graph.contract_random()

        # the final cut
        return local_graph

    @staticmethod
    def fast_cut(graph: Graph) -> Graph:
        """
        FastCut algorithm for minimum cut using rustworkx.

        Returns:
            nx.MultiGraph: THe smaller of the two cuts.
        """
        local_graph = graph.copy()

        def recursive_cut(subgraph: Graph):
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

        def contract_until(graph_to_contract: Graph, target_vertices: int) -> Graph:
            """Perform contractions until the graph has target_vertices left."""

            result = graph_to_contract.copy()
            while result.num_nodes() > target_vertices:
                result.contract_random()

            return result

        return recursive_cut(local_graph)

    @staticmethod
    def generate_random_graph_2(num_vertices, edge_probability=0.5) -> Graph:
        """Generates a random undirected graph using networkx."""
        result: Graph = Graph(dict(), list())
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
    GRAPH = Solver.generate_random_graph_2(10, 0.7)

    print(GRAPH)

    GRAPH.contract_random()
    print("AFTER : " + str(GRAPH))

    # Run Karger's algorithm
    # karger_cut = Solver.contract(GRAPH)
    # print("Karger's Algorithm Cut:", karger_cut)

    # print("Graph edges 2: ", graph.edges)

    # Run FastCut algorithm
    # fast_cut_result = Solver.fast_cut(GRAPH)
    # print("FastCut Algorithm Cut:", fast_cut_result)
