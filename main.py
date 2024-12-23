import random
from math import ceil, sqrt
from typing import Tuple, List
from dataclasses import dataclass
import time

"""
Description : Assignment for INFO-F413, Randomized algorithms, ULB.
Compare Karger’s Algorithm and the FastCut algorithm.

Author : Louis Devroye (523920) loudevroye@gmail.com
Date : 23/12/2024
VERSION: Python 3.12
LICENSE: MIT
"""


# random seed (my id)
SEED = 523920


@dataclass
class Graph:
    """
    a simple implementation of a possibly multigraph (several edges between two vertices)
    """
    edges: dict[int: List[int]]  # edges[1] = [2,3,5,8] edges between key and all the values

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
    def cut_size(self) -> int:
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

        return result // 2  # by the handshaking lemma (1 edge contribute to two keys) we can //2

    @property
    def nodes(self) -> List[int]:
        """
        :return: a list of the vertices
        """
        return list(self.edges.keys())

    @property
    def num_nodes(self) -> int:
        """
        :return: number of vertices in the graph
        """
        return len(self.edges.keys())

    def add_node(self, k=1):
        """
        Add 'k' vertices to the graph
        :param k: number of vertices to add
        """
        n: int = 0
        if self.num_nodes > 0:
            n = max(self.edges.keys()) + 1

        for i in range(n, n + k):
            self.edges[i] = []

    def add_edge(self, u: int, v: int):
        """
        Add an edge from u v (and from v to u since the graph is not directed)
        :param u: 1st vertex
        :param v: 2nd vertex
        """
        if u not in self.edges:
            self.edges[u] = []
        if v not in self.edges:
            self.edges[v] = []

        self.edges[u].append(v)
        self.edges[v].append(u)

    def remove_edge(self, u, v):
        """
        remove the edge between two vertices (in both sides since the graph is not directed)
        :param u: 1st vertex
        :param v: 2nd vertex
        """
        if u not in self.nodes and v not in self.nodes:
            raise ValueError(f"edge exception {u} or {v} not in graph")

        if v in self.edges[u]:
            self.edges[u].remove(v)
        if u in self.edges[v]:
            self.edges[v].remove(u)

    def get_random_edge(self) -> Tuple[int, int]:
        """
        :return: a tuple(u, v) corresponding to an edge
        """
        u = random.choice(self.nodes)
        v = random.choice(self.edges[u])
        return u, v

    def contract(self, u, v):
        """
        contracts two neighboring vertices by putting all of v's neighbors in u and then deleting v
        :param u: 1st vertex (the main target)
        :param v: 2nd vertex (the one deleted)
        """

        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Both nodes {u} and {v} must exist in the graph.")
        if u not in self.edges[v]:
            raise ValueError(f"{u}, {v} not neighbors")

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

    def contract_random_edge(self):
        """
        contract a random edge
        """
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
            int: the cut (nbr of edges) of the two vertices left
        """
        # the final cut
        return Solver.contract_until(graph, 2).cut_size

    @staticmethod
    def contract_until(graph_to_contract: Graph, target_vertices: int) -> Graph:
        """
        Perfom contractions on a copy of graph_to_contract until the graph has target_vertices left.
        :param graph_to_contract: the graph that will be copied
        :param target_vertices: nmber of vertices to have left after
        :return: the resulting graph
        """

        result = graph_to_contract.copy()
        while result.num_nodes > target_vertices:
            result.contract_random_edge()

        return result

    @staticmethod
    def fast_cut(graph: Graph, time_out: float = float('inf')) -> int:
        """
        FastCut algorithm for minimum cut. if the time out is exceeded, one last pass in the loop and the function
        returns the minimal cut computed

        :param graph: the graph that will be copied
        :param time_out: the time out (in secondes) after that the excectuion stops and returns the min_cut computed
        :Returns:
            int: The smallest cut.
        """
        local_graph: Graph = graph.copy()

        stack: List[Tuple[Graph, int]] = [(local_graph, local_graph.num_nodes)]
        min_cut = float('inf')

        begin: float = time.time()
        while stack and min_cut > 2 and (time.time() - begin < time_out):
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
            H1: Graph = Solver.contract_until(subgraph, t)
            H2: Graph = Solver.contract_until(subgraph, t)

            # c
            stack.append((H1, t))
            stack.append((H2, t))

        # d
        return min_cut

    @staticmethod
    def generate_random_graph(num_vertices, edge_probability=0.5) -> Graph:
        """
        Generate a partially random graph. This graph is 2-connected (if the implementation is correct,
        but it is at least simply connected because of the dfs() search). This avoids unstable and uninteresting graphs.
        :param num_vertices: the number of vertices the graph has
        :param edge_probability: the probability for all the edges to exists. (only one-sided,
        we add every edge (u, v) with u < v, except if the graph is not connected then we add random edges)

        :return: the created graph
        """
        result: Graph = Graph(dict())
        result.add_node(num_vertices)

        def dfs(node, visited):
            """embeded function to check connectivity of the graph"""
            visited.add(node)
            for neighbors in result.edges.values():
                for neighbor in neighbors:
                    if neighbor not in visited:
                        dfs(neighbor, visited)

        while True:
            for i in range(0, num_vertices):
                for j in range(i + 1, num_vertices):
                    if random.random() < edge_probability:
                        result.add_edge(i, j)

            for i in range(0, num_vertices):
                numbers = list(range(0, num_vertices))
                numbers.remove(i)
                random.shuffle(numbers)

                j = len(result.edges[i])
                if j == 1:  # don't take the same neighbor twice
                    numbers.remove(result.edges[i][0])

                while j < 2:
                    result.add_edge(i, numbers[j])
                    j += 1

            # check connectivity
            visited = set()
            start_node = result.nodes[random.randint(0, num_vertices-1)]
            dfs(start_node, visited)
            if len(visited) == result.num_nodes:
                break

        return result


def size_prob_test():
    sizes = [10, 50, 100]  # Different sizes of graphs
    probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different probabilities of edges

    results = []

    for size in sizes:
        for prob in probabilities:
            print(f"{size}, {prob} :")
            start_time = time.time()
            graph = Solver.generate_random_graph(size, prob)
            Solver.fast_cut(graph)
            elapsed_time = time.time() - start_time

            results.append({
                "size": size,
                "probability": prob,
                "time": elapsed_time,
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges
            })

            print(f"size {size} and probability {prob} in {elapsed_time:.6f} seconds.")

    return results


def main_test():
    print("Starting graph generation tests...")
    results = size_prob_test()

    print("\nSummary of Results:")
    for result in results:
        print(
            f"Size: {result['size']}, Probability: {result['probability']}, Time: {result['time']:.6f} sec, Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")


def main():
    #test parameters
    prob: float = 0
    num_v: int = 100
    number_try: int = 1000
    time_out: float = 10.0

    # printing purpose
    progress_threshold: int = 5
    update_interval: float = number_try * (progress_threshold / 100)  # Calculate steps per update

    # output
    num_edges = 0
    start_time = time.time()
    error = 0
    print(f"STARTING : {prob*100}%, {num_v} nodes, {number_try} iteration")
    for i in range(number_try):
        GRAPH = Solver.generate_random_graph(num_v, prob)
        try:
            Solver.fast_cut(GRAPH)
        except:
            error += 1
            print(f"ERROR: {GRAPH}")

        num_edges += GRAPH.num_edges

        if i % update_interval == 0 or i == number_try:
            progress = (i / number_try) * 100
            print(f"Progress: {progress+5:.0f}%")

    elapsed = time.time() - start_time
    print(f"Total time for {num_v} {prob*100}%: {elapsed:.4f} sec - {elapsed/number_try:.4f} sec average for {number_try} times")
    print(f"Summary of Results: {num_edges} edges - {num_edges/number_try} average")
    print(f"{error} errors ({error/number_try*100}%)")


def basic_main():

    prob: float = 0.5
    num_v = 100
    # Generate a random graph
    start_time = time.time()

    GRAPH = Solver.generate_random_graph(num_v, prob)
    print(f"STARTING {prob*100}%, {num_v} vertices, {GRAPH.num_edges} edges")

    # Run Karger's algorithm
    karger_cut = Solver.contract(GRAPH)
    second_time = time.time() - start_time

    print(f"Karger's Algorithm Cut: {karger_cut}")
    print(f"Time taken: {second_time:0.4f} seconds")

    # Run FastCut algorithm
    fast_cut_result = Solver.fast_cut(GRAPH, 120)
    third_time = time.time() - start_time

    print("FastCut Algorithm Cut:", fast_cut_result)
    print(f"Time taken: {third_time:0.4f} seconds")


if __name__ == "__main__":
    random.seed(SEED)
    basic_main()



