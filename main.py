import random
from math import ceil, sqrt
from typing import Tuple, Any, List, Set
from dataclasses import dataclass
import time


@dataclass
class Graph:
    edges: dict[int: List[int]]  # edges[1] = [2,3,5,8] connected nodes

    def __init__(self, edges: dict):
        self.edges: dict[int: List[int]] = edges

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return Graph({node: neighbors.copy() for node, neighbors in self.edges.items()})

    def __str__(self):
        str_edges = "{\n"
        for k in self.nodes:
            str_edges += f"\t{k}: {self.edges[k]}\n"
        str_edges += "\t}"

        return f"\nEdges: {str_edges}, {self.num_edges} \nNodes : {self.nodes}, {self.num_nodes}"

    @property
    def get_cut_size(self):
        return self.num_edges

    @property
    def num_edges(self) -> int:
        """
        returns the cut size (number of edges) of the graph
        :return: cut size
        """
        result: int = 0
        for k in self.edges.values():
            result += len(k)

        return result // 2

    @property
    def nodes(self) -> List[int]:
        return list(self.edges.keys())

    @property
    def num_nodes(self) -> int:
        return len(self.edges.keys())

    def add_node(self, k=1):
        n: int = 0
        if self.num_nodes > 0:
            n = max(self.edges.keys()) + 1

        for i in range(n, n + k):
            self.edges[i] = []

    def add_edge(self, u: int, v: int):
        if u not in self.edges:
            self.edges[u] = []
        if v not in self.edges:
            self.edges[v] = []

        self.edges[u].append(v)
        self.edges[v].append(u)

    def remove_edge(self, u, v):
        if u not in self.nodes and v not in self.nodes:
            raise ValueError(f"edge exception {u} or {v} not in graph")

        if v in self.edges[u]:
            self.edges[u].remove(v)
        if u in self.edges[v]:
            self.edges[v].remove(u)

    def get_random_edge(self) -> Tuple[int, int]:
        u = random.choice(self.nodes)
        v = random.choice(self.edges[u])
        return u, v

    def contract(self, u, v):
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Both nodes {u} and {v} must exist in the graph.")

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

    def contract_random(self):
        u, v = self.get_random_edge()
        #print("a: " + str(u) + ", b: " + str(v))
        #print(self.__str__())

        self.contract(u, v)



class Solver:

    @staticmethod
    def contract(graph: Graph) -> int:
        """
        Karger's Contract algorithm for minimum cut using rustworkx.

        Returns:
            int: the cut of the two vertices left
        """
        # the final cut
        return Solver.contract_until(graph, 2).num_edges

    @staticmethod
    def contract_until(graph_to_contract: Graph, target_vertices: int) -> Graph:
        """Perfom contractions on a copy of graph_to_contract until the graph has target_vertices left."""

        result = graph_to_contract.copy()
        while result.num_nodes > target_vertices:
            result.contract_random()

        return result


    @staticmethod
    def fast_cut(graph: Graph) -> int:
        """
        FastCut algorithm for minimum cut.

        Returns:
            int: The smallest cut.
        """
        local_graph = graph.copy()

        stack = [(local_graph, local_graph.num_nodes)]
        min_cut = float('inf')

        while stack and min_cut > 2:
            # 1.
            subgraph, n = stack.pop()

            # 2.
            if n <= 6:
                # Brute force: Run contract multiple times and take the best result
                min_cut = min(min_cut, Solver.contract(subgraph))
                continue

            # a
            t: int = ceil(1 + n / sqrt(2))

            # b
            H1 = Solver.contract_until(subgraph, t)
            H2 = Solver.contract_until(subgraph, t)

            # c
            stack.append((H1, t))
            stack.append((H2, t))

        # d
        return min_cut

    @staticmethod
    def generate_random_graph(num_vertices, edge_probability=0.5) -> Graph:
        """Generates a random undirected graph using networkx."""
        result: Graph = Graph(dict())
        result.add_node(num_vertices)

        for i in range(0, num_vertices):
            for j in range(i + 1, num_vertices):
                if random.random() < edge_probability:
                    result.add_edge(i, j)

        for i in range(0, num_vertices):
            numbers = list(range(0, num_vertices))
            numbers.remove(i)
            random.shuffle(numbers)
            j = len(result.edges[i])

            while j < 2:
                result.add_edge(i, numbers[j])

        return result


# Example usage
if __name__ == "__main__":
    random.seed(523920)

    prob: float = 0.1
    num_v = 100
    # Generate a random graph
    GRAPH = Solver.generate_random_graph(num_v, prob)

    print(f"STARTING {prob} {num_v}: {GRAPH}")

    #GRAPH.contract_random()
    #print("AFTER : " + str(GRAPH))

    start_time = time.time()

    # Run Karger's algorithm
    karger_cut = Solver.contract(GRAPH)
    print(f"Karger's Algorithm Cut: {karger_cut}")
    second_time = time.time() - start_time
    print("Time taken:", second_time, "seconds")

    # Run FastCut algorithm
    fast_cut_result = Solver.fast_cut(GRAPH)
    print("FastCut Algorithm Cut:", fast_cut_result)
    third_time = time.time()-start_time
    print("Time taken:", third_time, "seconds")
