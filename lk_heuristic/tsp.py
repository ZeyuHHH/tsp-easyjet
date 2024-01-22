import math
import logging
import random
from itertools import permutations
from lk_heuristic.edges import Edge
from lk_heuristic.tour import Tour
from lk_heuristic.nodes import NodePivot


class Tsp:
    gain_precision = 0.01

    def __init__(
        self,
        nodes,
        cost_function,
        shuffle=False,
        backtracking=(5, 5),
        reduction_level=4,
        reduction_cycle=4,
        tour_type="cycle",
        logging_level=logging.INFO,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
        self.nodes = nodes
        self.start_node = random.choice(nodes)
        self.tour_type = tour_type
        self.tour = Tour(self.nodes, t=tour_type)
        if shuffle:
            self.tour.shuffle()
        self.shuffle = True
        self.cost_matrix = {}
        self.set_cost_matrix(cost_function)
        self.tour.set_cost(self.cost_matrix)
        self.methods = {
            "bf_improve": self.bf_improve,
            "nn_improve": self.nn_improve,
            "lk1_improve": self.lk1_improve,
            "lk2_improve": self.lk2_improve,
        }
        self.solutions = set()
        self.closest_neighbors = {}
        self.set_closest_neighbors(max_neighbors=max(backtracking))
        self.backtracking = backtracking
        self.reduction_level = reduction_level
        self.reduction_cycle = reduction_cycle
        self.reduction_edges = set()
        self.cycles = 0
        self.close_gains = []
        self.best_close_gain = 0
        self.double_bridge_gain = 0

    def set_cost_matrix(self, cost_func):
        for i in range(len(self.nodes)):
            for j in range(i, len(self.nodes)):
                n1 = self.nodes[i]
                n2 = self.nodes[j]
                if type(n1) == NodePivot or type(n2) == NodePivot:
                    cost = 0
                else:
                    cost = cost_func(n1, n2)

                self.cost_matrix[(n1.id, n2.id)] = cost
                self.cost_matrix[(n2.id, n1.id)] = cost  # symmetric value

    def set_closest_neighbors(self, max_neighbors):
        for n1 in self.tour.nodes:
            neighbors = [
                (n2, self.cost_matrix[(n1.id, n2.id)])
                for n2 in self.tour.nodes
                if n2 != n1
            ]
            neighbors_min = sorted(neighbors, key=lambda x: x[1])[:max_neighbors]
            self.closest_neighbors[n1] = [neighbor[0] for neighbor in neighbors_min]

    def get_best_neighbors(self, t2, t1=None):
        best_neighbors = {}

        for t3 in self.closest_neighbors[t2]:
            for t4 in (t3.pred, t3.succ):
                if t1:
                    if self.tour.is_swap_feasible(t1, t2, t3, t4):
                        best_neighbors[(t3, t4)] = (
                            self.cost_matrix[(t3.id, t4.id)]
                            - self.cost_matrix[t2.id, t3.id]
                        )
                else:
                    best_neighbors[(t3, t4)] = (
                        self.cost_matrix[(t3.id, t4.id)]
                        - self.cost_matrix[t2.id, t3.id]
                    )
        return sorted(best_neighbors.items(), key=lambda x: x[1], reverse=True)

    def lk1_feasible_search(
        self, level, gain, swap_func, t1, t2, t3, t4, broken_edges, joined_edges
    ):
        broken_edge = Edge(t3, t4)
        broken_cost = self.cost_matrix[(t3.id, t4.id)]
        if (
            level >= self.reduction_level
            and self.cycles <= self.reduction_cycle
            and broken_edge in self.reduction_edges
        ):
            return
        broken_edges.add(Edge(t1, t2))
        joined_edges.add(Edge(t2, t3))
        if swap_func == "swap_feasible":
            self.tour.swap_feasible(t1, t2, t3, t4)
        elif swap_func == "swap_node_between_t2_t3":
            self.tour.swap_node_between_t2_t3(t1, t2, t3, t4)
        joined_close_edge = Edge(t4, t1)
        joined_close_cost = self.cost_matrix[(t4.id, t1.id)]
        joined_close_valid = (
            joined_close_edge not in self.tour.edges
            and joined_close_edge not in broken_edges
        )
        close_gain = gain + (broken_cost - joined_close_cost)
        self.close_gains.append(close_gain)
        self.best_close_gain = (
            close_gain if close_gain > self.best_close_gain else self.best_close_gain
        )
        curr_backtracking = 1
        if level <= len(self.backtracking) - 1:
            curr_backtracking = self.backtracking[level]
        for (next_y_head, next_x_head), _ in self.get_best_neighbors(t4, t1)[
            :curr_backtracking
        ]:
            joined_edge = Edge(t4, next_y_head)
            joined_cost = self.cost_matrix[(t4.id, next_y_head.id)]
            explore_gain = gain + (broken_cost - joined_cost)
            disjoint_criteria = False
            if broken_edge not in broken_edges and broken_edge not in joined_edges:
                if (
                    joined_edge not in self.tour.edges
                    and joined_edge not in broken_edges
                ):
                    disjoint_criteria = True
            gain_criteria = False
            if explore_gain > self.gain_precision:
                gain_criteria = True
            next_xi_criteria = False
            next_broken_edge = Edge(next_y_head, next_x_head)
            if (
                next_broken_edge not in broken_edges
                and next_broken_edge not in joined_edges
            ):
                next_xi_criteria = True
            if disjoint_criteria and gain_criteria and next_xi_criteria:
                if (
                    hash(tuple([node.succ.id for node in self.tour.nodes]))
                    in self.solutions
                ):
                    return
                if (
                    close_gain > explore_gain
                    and close_gain >= self.best_close_gain
                    and close_gain > self.gain_precision
                    and joined_close_valid
                ):
                    broken_edges.add(broken_edge)
                    joined_edges.add(joined_close_edge)
                    return
                else:
                    return self.lk1_feasible_search(
                        level + 1,
                        explore_gain,
                        "swap_feasible",
                        t1,
                        t4,
                        next_y_head,
                        next_x_head,
                        broken_edges,
                        joined_edges,
                    )

    def lk1_unfeasible_search(self, gain, t1, t2, t3, t4, broken_edges, joined_edges):
        broken_edges.add(Edge(t1, t2))
        joined_edges.add(Edge(t2, t3))
        broken_edge_1 = Edge(t3, t4)
        broken_cost_1 = self.cost_matrix[(t3.id, t4.id)]
        self.tour.swap_unfeasible(t1, t2, t3, t4)
        self.close_gains.append(-1)
        curr_backtracking = 1
        if len(self.backtracking) - 1 >= 1:
            curr_backtracking = self.backtracking[1]
        for (t5, t6), _ in self.get_best_neighbors(t4)[:curr_backtracking]:
            joined_edge_1 = Edge(t4, t5)
            joined_cost_1 = self.cost_matrix[(t4.id, t5.id)]
            explore_gain = gain + (broken_cost_1 - joined_cost_1)
            gain_criteria = False
            if explore_gain > self.gain_precision:
                gain_criteria = True
            valid_nodes = False
            if t5 != t1 and t5 != t2 and t5 != t3 and t5 != t4:
                if t6 != t1 and t6 != t2 and t6 != t3 and t6 != t4:
                    valid_nodes = True
            if valid_nodes and gain_criteria:
                broken_edge_2 = Edge(t5, t6)
                broken_cost_2 = self.cost_matrix[(t5.id, t6.id)]
                t5_between_t1_t4 = False
                t1_after_t4 = t4.succ == t1
                if t1_after_t4:
                    t5_between_t1_t4 = self.tour.between(t1, t5, t4)
                else:
                    t5_between_t1_t4 = self.tour.between(t4, t5, t1)
                if t5_between_t1_t4:
                    if self.tour.is_swap_feasible(t1, t4, t5, t6):
                        curr_backtracking = 1
                        if len(self.backtracking) - 1 >= 2:
                            curr_backtracking = self.backtracking[2]
                        for (t7, t8), _ in self.get_best_neighbors(t6)[
                            :curr_backtracking
                        ]:
                            joined_edge_2 = Edge(t6, t7)
                            joined_cost_2 = self.cost_matrix[(t6.id, t7.id)]
                            explore_gain += broken_cost_2 - joined_cost_2
                            gain_criteria = False
                            if explore_gain > self.gain_precision:
                                gain_criteria = True
                            t7_between_t2_t3 = False
                            t2_after_t3 = t3.succ == t2
                            if t2_after_t3:
                                t7_between_t2_t3 = self.tour.between(t2, t7, t3)
                            else:
                                t7_between_t2_t3 = self.tour.between(t3, t7, t2)
                            valid_nodes = False
                            if t7 != t2 and t7 != t3 and t8 != t2 and t8 != t3:
                                valid_nodes = True
                            if gain_criteria and valid_nodes and t7_between_t2_t3:
                                broken_edge_3 = Edge(t7, t8)
                                self.tour.swap_feasible(t1, t4, t5, t6, is_subtour=True)
                                self.close_gains.append(-1)
                                broken_edges.add(broken_edge_1)
                                broken_edges.add(broken_edge_2)
                                broken_edges.add(broken_edge_3)
                                joined_edges.add(joined_edge_1)
                                joined_edges.add(joined_edge_2)
                                return self.lk1_feasible_search(
                                    4,
                                    explore_gain,
                                    "swap_node_between_t2_t3",
                                    t1,
                                    t6,
                                    t7,
                                    t8,
                                    broken_edges,
                                    joined_edges,
                                )
                else:
                    broken_edges.add(broken_edge_1)
                    broken_edges.add(broken_edge_2)
                    joined_edges.add(joined_edge_1)
                    return self.lk1_feasible_search(
                        3,
                        explore_gain,
                        "swap_node_between_t2_t3",
                        t1,
                        t4,
                        t5,
                        t6,
                        broken_edges,
                        joined_edges,
                    )
        self.tour.restore()

    def lk1_double_bridge_search(self, max_tests=100):
        search_edges = list(self.tour.edges.difference(self.reduction_edges))
        if len(search_edges) >= 4:
            for _ in range(max_tests):
                random.shuffle(search_edges)
                broken_edge_1 = search_edges[0]
                broken_edge_2 = search_edges[1]
                broken_edge_3 = search_edges[2]
                broken_edge_4 = search_edges[3]
                double_bridge_nodes = self.tour.is_swap_double_bridge(
                    broken_edge_1.n1,
                    broken_edge_1.n2,
                    broken_edge_2.n1,
                    broken_edge_2.n2,
                    broken_edge_3.n1,
                    broken_edge_3.n2,
                    broken_edge_4.n1,
                    broken_edge_4.n2,
                )
                if double_bridge_nodes:
                    t1 = double_bridge_nodes[0]
                    t2 = double_bridge_nodes[1]
                    t3 = double_bridge_nodes[2]
                    t4 = double_bridge_nodes[3]
                    t5 = double_bridge_nodes[4]
                    t6 = double_bridge_nodes[5]
                    t7 = double_bridge_nodes[6]
                    t8 = double_bridge_nodes[7]
                    broken_cost_1 = self.cost_matrix[(t1.id, t2.id)]
                    broken_cost_2 = self.cost_matrix[(t3.id, t4.id)]
                    broken_cost_3 = self.cost_matrix[(t5.id, t6.id)]
                    broken_cost_4 = self.cost_matrix[(t7.id, t8.id)]
                    joined_cost_1 = self.cost_matrix[(t1.id, t4.id)]
                    joined_cost_2 = self.cost_matrix[(t2.id, t3.id)]
                    joined_cost_3 = self.cost_matrix[(t5.id, t8.id)]
                    joined_cost_4 = self.cost_matrix[(t6.id, t7.id)]
                    gain = (
                        broken_cost_1 + broken_cost_2 + broken_cost_3 + broken_cost_4
                    ) - (joined_cost_1 + joined_cost_2 + joined_cost_3 + joined_cost_4)
                    if gain > self.gain_precision:
                        self.tour.swap_double_bridge(
                            t1, t2, t3, t4, t5, t6, t7, t8, False
                        )
                        self.double_bridge_gain = gain
                        self.tour.edges.remove(Edge(t1, t2))
                        self.tour.edges.remove(Edge(t3, t4))
                        self.tour.edges.remove(Edge(t5, t6))
                        self.tour.edges.remove(Edge(t7, t8))
                        self.tour.edges.add(Edge(t1, t4))
                        self.tour.edges.add(Edge(t2, t3))
                        self.tour.edges.add(Edge(t5, t8))
                        self.tour.edges.add(Edge(t6, t7))
                        break

    def lk1_main(self):
        for t1 in self.tour.get_nodes(start_node=self.start_node):
            for t2 in (t1.pred, t1.succ):
                broken_edge = Edge(t1, t2)
                broken_cost = self.cost_matrix[(t1.id, t2.id)]
                for (t3, t4), _ in self.get_best_neighbors(t2)[: self.backtracking[0]]:
                    joined_edge = Edge(t3, t2)
                    joined_cost = self.cost_matrix[(t3.id, t2.id)]
                    gain = broken_cost - joined_cost
                    if (
                        joined_edge not in self.tour.edges
                        and gain > self.gain_precision
                    ):
                        broken_edges = set()
                        joined_edges = set()
                        if self.tour.is_swap_feasible(t1, t2, t3, t4):
                            self.lk1_feasible_search(
                                1,
                                gain,
                                "swap_feasible",
                                t1,
                                t2,
                                t3,
                                t4,
                                broken_edges,
                                joined_edges,
                            )
                        elif self.tour.is_swap_unfeasible(t1, t2, t3, t4):
                            self.lk1_unfeasible_search(
                                gain, t1, t2, t3, t4, broken_edges, joined_edges
                            )
                        if self.close_gains:
                            if max(self.close_gains) > 0:
                                best_index = self.close_gains.index(
                                    max(self.close_gains)
                                )
                                for i in range(best_index + 1):
                                    (n1, n2, n3, n4, _) = self.tour.swap_stack[i]
                                    self.tour.edges.remove(Edge(n1, n2))
                                    self.tour.edges.remove(Edge(n3, n4))
                                    self.tour.edges.add(Edge(n2, n3))
                                    self.tour.edges.add(Edge(n4, n1))
                                self.tour.restore(
                                    (len(self.close_gains) - 1) - best_index
                                )
                                self.start_node = self.tour.swap_stack[-1][3]
                                self.tour.set_cost(self.cost_matrix)
                                self.close_gains.clear()
                                return True

                            else:
                                self.close_gains.clear()
                                self.tour.restore()
        return False

    def lk1_improve(self):
        tour_count = 1
        improved = True
        self.logger.debug(f"Starting tour cost: {self.tour.cost:.3f}")
        while improved:
            improved = self.lk1_main()
            total_swaps = len(self.tour.swap_stack)
            feasible_swaps = len(
                [swap for swap in self.tour.swap_stack if swap[-1] == "swap_feasible"]
            )
            unfeasible_swaps = len(
                [swap for swap in self.tour.swap_stack if swap[-1] != "swap_feasible"]
            )
            self.logger.debug(
                f"Current tour '{tour_count}' cost: {self.tour.cost:.3f} / gain: {self.best_close_gain:.3f} / swaps: {total_swaps} / feasible swaps: {feasible_swaps} / unfeasible swaps: {unfeasible_swaps}"
            )
            tour_count += 1
            self.tour.swap_stack.clear()
            self.best_close_gain = 0
        self.cycles += 1
        self.solutions.add(hash(tuple([node.succ.id for node in self.tour.nodes])))
        self.solutions.add(hash(tuple([node.pred.id for node in self.tour.nodes])))
        self.reduction_edges = (
            set(self.tour.edges)
            if self.cycles == 1
            else self.reduction_edges.intersection(self.tour.edges)
        )
        if self.cycles >= self.reduction_cycle:
            self.lk1_double_bridge_search()
            if self.double_bridge_gain > 0:
                self.logger.info(
                    f"Double bridge move found: cost: {self.tour.cost:.3f} / gain: {self.double_bridge_gain:.3f}"
                )
                self.double_bridge_gain = 0
                self.shuffle = False
            else:
                self.shuffle = True

    def lk2_select_broken_edge(self, gain, t1, t2, t3, t4, broken_edges, joined_edges):
        broken_edge = Edge(t3, t4)
        broken_cost = self.cost_matrix[(t3.id, t4.id)]
        if (
            t1 != t4
            and broken_edge not in joined_edges
            and broken_edge not in broken_edges
        ):
            if self.tour.is_swap_feasible(t1, t2, t3, t4):
                self.tour.swap_feasible(t1, t2, t3, t4)
                broken_edges.add(broken_edge)
                joined_edge = Edge(t4, t1)
                joined_cost = self.cost_matrix[(t4.id, t1.id)]
                curr_gain = gain + (broken_cost - joined_cost)
                if curr_gain > self.gain_precision:
                    joined_edges.add(joined_edge)
                    self.tour.set_cost(self.cost_matrix)
                    return True
                else:
                    return self.lk2_select_joined_edge(
                        curr_gain, t1, t4, broken_edges, joined_edges
                    )
        return False

    def lk2_select_joined_edge(self, gain, t1, t4, broken_edges, joined_edges):
        broken_cost = self.cost_matrix[(t4.id, t1.id)]
        for (node, neighbor_node), _ in self.get_best_neighbors(t4, t1):
            joined_edge = Edge(t4, node)
            joined_cost = self.cost_matrix[(t4.id, node.id)]
            curr_gain = gain + (broken_cost - joined_cost)
            if (
                joined_edge not in broken_edges
                and joined_edge not in self.tour.edges
                and curr_gain > self.gain_precision
            ):
                joined_edges.add(joined_edge)
                return self.lk2_select_broken_edge(
                    curr_gain, t1, t4, node, neighbor_node, broken_edges, joined_edges
                )
        return False

    def lk2_main(self):
        for t1 in self.tour.get_nodes(start_node=self.start_node):
            for t2 in (t1.pred, t1.succ):
                broken_edge = Edge(t1, t2)
                broken_cost = self.cost_matrix[(t1.id, t2.id)]
                for (t3, t4), _ in self.get_best_neighbors(t2, t1):
                    joined_edge = Edge(t3, t2)
                    joined_cost = self.cost_matrix[(t3.id, t2.id)]
                    gain = broken_cost - joined_cost
                    if (
                        joined_edge not in self.tour.edges
                        and gain > self.gain_precision
                    ):
                        broken_edges = set([broken_edge])
                        joined_edges = set([joined_edge])
                        if self.lk2_select_broken_edge(
                            gain, t1, t2, t3, t4, broken_edges, joined_edges
                        ):
                            self.tour.swap_stack.clear()
                            self.start_node = t4
                            return True
                        else:
                            self.tour.restore()
        return False

    def lk2_improve(self):
        tour_count = 1
        improved = True
        self.logger.debug(f"Starting tour cost: {self.tour.cost:.3f}")
        while improved:
            improved = self.lk2_main()
            self.logger.debug(f"Current tour '{tour_count}' cost: {self.tour.cost:.3f}")
            tour_count += 1

    def bf_improve(self):
        min_cost = math.inf
        min_tour = None
        perms = permutations(self.nodes)
        start_node = self.nodes[0]
        tour_count = 0

        # loop through each permutation
        for perm in perms:
            # only permutation starting with the starting node is considered (so that repeated tours starting at a different node are not considered)
            if perm[0] == start_node:
                # loop through each node (starting from the last one)
                for i in range(-1, len(perm) - 1):
                    # get current and next nodes
                    curr_node = perm[i]
                    next_node = perm[i + 1]

                    # assign predecessor and successor nodes
                    curr_node.succ = next_node
                    next_node.pred = curr_node

                # update the cost value
                self.tour.set_cost(self.cost_matrix)

                # check if a best tour was found
                if self.tour.cost < min_cost:
                    # update best values
                    min_cost = self.tour.cost
                    min_tour = perm

                    # log the current  tour cost
                    self.logger.info(
                        f"Current tour '{tour_count}'cost: {self.tour.cost:.3f}"
                    )

                # update and log count
                tour_count += 1
                if tour_count % 1000 == 0:
                    self.logger.info(
                        f"Current tour '{tour_count}' cost: {min_cost:.3f}"
                    )

        # loop through each node of the best tour
        for i in range(-1, len(min_tour) - 1):
            # get current and next nodes
            curr_node = min_tour[i]
            next_node = min_tour[i + 1]

            # assign predecessor and successor nodes
            curr_node.succ = next_node
            next_node.pred = curr_node

        # update the cost value
        self.tour.set_cost(self.cost_matrix)

    def nn_improve(self):
        """
        The improve loop using Nearest-Neighbor algorithm.
        """

        # create a set of all tsp nodes
        nodes = set(self.nodes)

        # get the starting node
        start_node = random.choice(self.nodes)
        curr_node = start_node

        # start the set of visited nodes
        visited_nodes = set()
        visited_nodes.add(curr_node)

        # loop until all nodes are visited
        while len(visited_nodes) < len(self.nodes):
            cost = math.inf

            # loop through each node that were not visited yet
            for node in nodes - visited_nodes:
                if self.cost_matrix[(node.id, curr_node.id)] < cost:
                    cost = self.cost_matrix[(node.id, curr_node.id)]
                    next_node = node

            # update succ and pred for current and next nodes
            next_node.pred = curr_node
            curr_node.succ = next_node

            # add new node to visited nodes set
            visited_nodes.add(next_node)

            # update current node for next iteration
            curr_node = next_node

        # update the starting and ending nodes
        start_node.pred = curr_node
        curr_node.succ = start_node

        # update the cost value
        self.tour.set_cost(self.cost_matrix)

    @classmethod
    def get_solution_methods(cls):
        return [cls.bf_improve, cls.nn_improve, cls.lk1_improve, cls.lk2_improve]
