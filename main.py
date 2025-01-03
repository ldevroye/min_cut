import random
from math import ceil, sqrt
from typing import Tuple, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
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
        # print("a: " + str(u) + ", b: " + str(v))
        # print(self.__str__())
        self.contract(u, v)


class Solver:

    @staticmethod
    def contract(graph: Graph, time_out: float = float('inf')) -> int:
        """
        Karger's Contract algorithm for minimum cut using rustworkx.

        Returns:
            int: the cut (nbr of edges) of the two vertices left
        """
        # the final cut
        return Solver.contract_until(graph, 2, time_out).cut_size

    @staticmethod
    def contract_until(graph_to_contract: Graph, target_vertices: int, time_out: float = float('inf')) -> Graph:
        """
        Perfom contractions on a copy of graph_to_contract until the graph has target_vertices left.
        :param time_out: time before returning the result
        :param graph_to_contract: the graph that will be copied
        :param target_vertices: nmber of vertices to have left after
        :return: the resulting graph
        """
        start: float = time.time()
        result: Graph = graph_to_contract.copy()
        while result.num_nodes > target_vertices and time.time()-start < time_out:
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

        stack: List[Graph] = [local_graph]
        min_cut = float('inf')

        begin: float = time.time()
        while stack and min_cut > 2 and (time.time() - begin < time_out):
            # 1.
            subgraph = stack.pop()
            n = subgraph.num_nodes
            # 2.
            if n <= 6:
                # Brute force: Run contract multiple times and take the best result
                min_cut = min(min_cut, Solver.contract(subgraph))
                continue

            # a
            t: int = ceil(1 + n / sqrt(2))

            # b
            remain_time: float = time_out-(time.time()-begin)
            H1: Graph = Solver.contract_until(subgraph, t, remain_time)
            H2: Graph = Solver.contract_until(subgraph, t, remain_time)

            # c
            stack.append(H1)
            stack.append(H2)

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

        def dfs(node: int, already_visited: set):
            """embeded iterative function to check connectivity of the graph"""
            stack = [node]

            while stack:
                current_node = stack.pop()
                if current_node not in already_visited:
                    already_visited.add(current_node)
                    # Add all unvisited neighbors to the stack
                    for neighbor in result.edges[current_node]:
                        if neighbor not in already_visited:
                            stack.append(neighbor)

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
            start_node = result.nodes[random.randint(0, num_vertices - 1)]
            dfs(start_node, visited)
            if len(visited) == result.num_nodes:
                break

        return result


def test(density: float, num_vertices: list[int], num_iteration: int = 1, time_out: float = float('inf')) -> dict:
    """
    Try to do all densities for all the num_v vertices for num_iteration times with time_out time for each iteration
    :param density: densities (probability for every edges to happen) of the graphs from 0 to 1
    :param num_vertices: List of vertices from 0 to inf
    :param num_iteration: number of iteration
    :param time_out: time out for each algorithm iteration
    :return: a dictionnary with dict[number_vertices] = [(cut_contract, cut_fast),
                                                        (error_contract, error_fast),
                                                        (compute_time_contract, compute_time_fast)]
    """
    # printing purpose
    progress_threshold: int = 5
    update_interval: float = num_iteration * (progress_threshold / 100)  # Calculate steps per update

    # output
    output: dict[int: List[tuple[int, int]]] = dict()
    print(f"STARTING : {density} densitites, {num_vertices} nodes, {num_iteration} iteration")

    for n_v in num_vertices:
        num_edges: int = 0
        error: list[int] = [0, 0]
        minimal_cut: list[int] = [0, 0]
        computation_time: list[float] = [0, 0]

        start_time = time.time()

        print(n_v, density, num_iteration)

        for i in range(num_iteration):
            GRAPH = Solver.generate_random_graph(n_v, density)
            try:
                start_try = time.time()
                minimal_cut[0] += Solver.contract(GRAPH, time_out)
                computation_time[0] += time.time()-start_try
            except:
                error[0] += 1

            try:
                start_try = time.time()
                minimal_cut[1] += Solver.fast_cut(GRAPH, time_out)
                computation_time[1] += time.time()-start_try

            except:
                error[1] += 1

            num_edges += GRAPH.num_edges

            if i % update_interval == 0 or i == num_iteration and len(num_vertices) == 1:
                progress = (i / num_iteration) * 100
                #print(f"Progress: {progress + 5:.0f}%")

        elapsed = time.time() - start_time
        print(f"\tTotal time: {elapsed:.4f} sec - {elapsed / num_iteration:.4f} sec average"
              f" ({computation_time[0] / num_iteration:.4f}, {computation_time[1] / num_iteration:.4f})")
        print(f"\tSummary of Results: {num_edges} edges - {num_edges / num_iteration} average")
        print(f"\t{error} errors ({error[0] / num_iteration * 100:0.2f}%, {error[1] / num_iteration * 100:0.2f}%)")
        print(f"\t({minimal_cut[0] // num_iteration}, {minimal_cut[1] // num_iteration}) minimal cut")


        # minimal cut, error, computation time
        output[n_v] = [(minimal_cut[0] // num_iteration, minimal_cut[1] // num_iteration),
                       (error[0] / num_iteration * 100, error[1] / num_iteration * 100),
                       (computation_time[0] / num_iteration, computation_time[1] / num_iteration)]
    return output


def main_test():
    random.seed(SEED)
    density: float = 0.05
    num_v: List[int] = [10 * (i + 1) for i in range(0, 10)]
    num_try: int = 100
    set_time_out: float = 18

    result = test(density, num_v, num_try, set_time_out)

    error_dic: dict[int: tuple[int, int]] = dict()
    cut_dic: dict[int: tuple[int, int]] = dict()
    computation_time_dic: dict[int: tuple[int, int]] = dict()

    for key, value in result.items():
        # Unpack each tuple into separate lists
        cut_dic[key] = value[0]
        error_dic[key] = value[1]
        computation_time_dic[key] = value[2]

    x = "Number of Nodes"
    title = f"Contract vs FastCut Algorithms - {density * 100:0.0f}% density"

    y = "Error rate"
    title_1 = f"{y} - " + title
    plot_algorithm(error_dic, x, y, title_1)

    y = "Minimal cut "
    title_2 = f"{y} - " + title
    plot_algorithm(cut_dic, x, y, title_2)

    y = "Computation time (s)"
    title_3 = f"{y} - " + title
    plot_algorithm(computation_time_dic, x, y, title_3)


def plot_algorithm(dic: dict[int: List[Tuple]], x_label: str, y_label: str, title_label: str):
    """
    Plots Contract and FastCut algorithms comparisons.

    Parameters:
    - dic: [parameter]: (result_contract, result_fast_cut)
    - x_label: label to put in x
    - y_label: label to put in y
    """
    plt.figure(figsize=(10, 6))

    nodes: list = list(dic.keys())
    contract: list = []
    fast_cut: list = []
    for e in dic.values():
        contract.append(e[0])
        fast_cut.append(e[1])

    # Plot Contract algorithm runtime
    plt.plot(nodes, contract, label="Contract Algorithm", marker='o', color='blue')

    # Plot FastCut algorithm runtime
    plt.plot(nodes, fast_cut, label="FastCut Algorithm", marker='s', color='green')

    # Add labels, title, and legend
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title_label, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()


def basic_main():
    """
    easy main for 1 graph showing everything going on
    """

    prob: float = 0.05
    num_v = 200
    time_out = 10
    # Generate a random graph
    start_time = time.time()

    GRAPH = Solver.generate_random_graph(num_v, prob)
    print(GRAPH)
    print(f"STARTING {prob * 100:0.0f}%, {num_v} vertices, {GRAPH.num_edges} edges")

    # Run Karger's algorithm
    karger_cut = Solver.contract(GRAPH, time_out)
    second_time = time.time() - start_time

    print(f"Karger's Algorithm Cut: {karger_cut}")
    print(f"Time taken: {second_time:0.3f} seconds")

    # Run FastCut algorithm
    fast_cut_result = Solver.fast_cut(GRAPH, time_out)
    third_time = time.time() - start_time

    print("FastCut Algorithm Cut:", fast_cut_result)
    print(f"Time taken: {third_time:0.3f} seconds")


if __name__ == "__main__":
    #basic_main()
    main_test()
