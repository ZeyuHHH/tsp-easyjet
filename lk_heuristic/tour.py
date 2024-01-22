from random import shuffle, choice
from lk_heuristic.nodes import NodePivot
from lk_heuristic.edges import Edge


class Tour:
    def __init__(self, nodes, t="cycle"):
        self.t = t
        self.nodes = nodes
        self.set_nodes()
        self.edges = set()
        self.set_edges()
        self.cost = 0
        self.size = len(self.nodes)
        self.swap_stack = []

    def set_nodes(self):
        if self.t == "path":
            self.nodes.append(NodePivot())

        for i in range(len(self.nodes)):
            if i == len(self.nodes) - 1:
                self.nodes[i].succ = self.nodes[0]
            else:
                self.nodes[i].succ = self.nodes[i + 1]
            self.nodes[i].pred = self.nodes[i - 1]

            self.nodes[i].pos = i
            self.nodes[i].id = i

    def set_edges(self):
        tour_edges = set()

        curr_node = self.nodes[0]

        while curr_node.succ != self.nodes[0]:
            tour_edges.add(Edge(curr_node, curr_node.succ))
            curr_node = curr_node.succ

        tour_edges.add(Edge(curr_node, curr_node.succ))

        self.edges = tour_edges

    def set_cost(self, cost_matrix):
        tour_cost = 0

        curr_node = self.nodes[0]

        while curr_node.succ != self.nodes[0]:
            tour_cost += cost_matrix[(curr_node.id, curr_node.succ.id)]
            curr_node = curr_node.succ

        tour_cost += cost_matrix[(curr_node.id, curr_node.succ.id)]

        self.cost = tour_cost

    def set_pos(self):
        curr_node = self.nodes[0]
        curr_node.pos = 0

        while curr_node.succ != self.nodes[0]:
            curr_node = curr_node.succ
            curr_node.pos = curr_node.pred.pos + 1

    def get_nodes(self, random_start=False, start_node=None):
        visited_nodes = set(self.nodes)

        tour_nodes = []

        curr_node = self.nodes[0]
        if start_node:
            curr_node = start_node
        elif random_start:
            curr_node = choice(self.nodes)
        elif self.t == "path":
            for node in self.nodes:
                if type(node) == NodePivot:
                    curr_node = node
                    break

        visited_nodes.remove(curr_node)
        while len(visited_nodes) > 0:
            while curr_node.succ in visited_nodes:
                tour_nodes.append(curr_node)

                curr_node = curr_node.succ

                visited_nodes.remove(curr_node)
            tour_nodes.append(curr_node)
            if len(visited_nodes) > 0:
                curr_node = visited_nodes.pop()

        return tour_nodes

    def shuffle(self):
        indexes = [i for i in range(self.size)]
        shuffle(indexes)

        curr_node = self.nodes[indexes[-1]]

        for i in range(-1, self.size - 1):
            curr_node.succ = self.nodes[indexes[i + 1]]
            curr_node.pred = self.nodes[indexes[i - 1]]
            curr_node.pos = i + 1
            curr_node = curr_node.succ

        self.set_edges()

    def restore(self, swaps=None):
        if swaps == None:
            swaps = len(self.swap_stack)

        for _ in range(swaps):
            curr_stack = self.swap_stack[-1]

            t1 = curr_stack[0]
            t2 = curr_stack[1]
            t3 = curr_stack[2]
            t4 = curr_stack[3]

            swap_type = curr_stack[-1]
            if swap_type == "swap_feasible":
                self.swap_feasible(t4, t1, t2, t3, False, False)
            elif swap_type == "swap_unfeasible":
                self.swap_unfeasible(t4, t1, t2, t3, False, False)
            elif swap_type == "swap_node_between_t2_t3":
                self.swap_unfeasible(t4, t1, t2, t3, False, False)
            elif swap_type == "swap_node_between_t2_t3_reversed":
                self.swap_unfeasible(t4, t1, t2, t3, True, False)
            elif swap_type == "swap_feasible_reversed":
                self.swap_feasible(t4, t1, t2, t3, True, False)

            self.swap_stack.pop()
        for swap in self.swap_stack:
            if swap[-1] != "swap_feasible":
                self.set_pos()
                break

    def between(self, from_node, between_node, to_node, use_pos_attr=False):
        if use_pos_attr:
            if from_node.pos <= to_node.pos:
                return (
                    from_node.pos < between_node.pos and between_node.pos < to_node.pos
                )
            else:
                return (
                    from_node.pos < between_node.pos or between_node.pos < to_node.pos
                )
        else:
            node = from_node.succ

            while node != to_node:
                if node == between_node:
                    return True
                else:
                    node = node.succ

            return False

    def is_swap_feasible(self, t1, t2, t3, t4):
        if not (
            t1 != t2 and t1 != t3 and t1 != t4 and t2 != t3 and t2 != t4 and t3 != t4
        ):
            return False
        if t1.succ == t2:
            if t4 != t3.pred:
                return False
        elif t1.pred == t2:
            if t4 != t3.succ:
                return False

        return True

    def is_swap_unfeasible(self, t1, t2, t3, t4):
        if not (
            t1 != t2 and t1 != t3 and t1 != t4 and t2 != t3 and t2 != t4 and t3 != t4
        ):
            return False
        if t1.succ == t2:
            if t4 == t3.pred:
                return False
        elif t1.pred == t2:
            if t4 == t3.succ:
                return False
        if t2.pred == t3 or t2.succ == t3 or t1.pred == t4 or t1.succ == t4:
            return False

        return True

    def is_swap_double_bridge(self, t1, t2, t3, t4, t5, t6, t7, t8):
        if not (
            t1 != t3
            and t1 != t4
            and t1 != t5
            and t1 != t6
            and t1 != t7
            and t1 != t8
            and t2 != t3
            and t2 != t4
            and t2 != t5
            and t2 != t6
            and t2 != t7
            and t2 != t8
            and t3 != t5
            and t3 != t6
            and t3 != t7
            and t3 != t8
            and t4 != t5
            and t4 != t6
            and t4 != t7
            and t4 != t8
            and t5 != t7
            and t5 != t8
            and t6 != t7
            and t6 != t8
        ):
            return None
        if t1.pred == t2:
            temp = t2
            t2 = t1
            t1 = temp
        if t3.pred == t4:
            temp = t4
            t4 = t3
            t3 = temp
        if t5.pred == t6:
            temp = t6
            t6 = t5
            t5 = temp
        if t7.pred == t8:
            temp = t8
            t8 = t7
            t7 = temp
        nodes = sorted((t1, t3, t5, t7), key=lambda el: el.pos)
        return (
            nodes[0],
            nodes[0].succ,
            nodes[2],
            nodes[2].succ,
            nodes[1],
            nodes[1].succ,
            nodes[3],
            nodes[3].succ,
        )

    def swap_feasible(self, t1, t2, t3, t4, is_subtour=False, record=True):
        if t1.succ != t2:
            temp = t1
            t1 = t2
            t2 = temp
            temp = t3
            t3 = t4
            t4 = temp
        seg_size = t2.pos - t3.pos
        if seg_size < 0:
            seg_size += self.size
        if 2 * seg_size > self.size:
            temp = t3
            t3 = t2
            t2 = temp
            temp = t4
            t4 = t1
            t1 = temp
        pos = t1.pos
        node = t3
        end_node = t1.succ
        while node != end_node:
            temp = node.succ
            node.succ = node.pred
            node.pred = temp
            if not is_subtour:
                node.pos = pos
                pos -= 1
            node = temp
        t3.succ = t2
        t2.pred = t3
        t1.pred = t4
        t4.succ = t1
        if record:
            if not is_subtour:
                self.swap_stack.append((t1, t2, t3, t4, "swap_feasible"))
            else:
                self.swap_stack.append((t1, t2, t3, t4, "swap_feasible_reversed"))

    def swap_unfeasible(self, t1, t2, t3, t4, reverse_subtour=False, record=True):
        if t1.succ == t2:
            temp = t3
            t3 = t2
            t2 = temp
            temp = t4
            t4 = t1
            t1 = temp
        t3.pred = t2
        t2.succ = t3
        t1.pred = t4
        t4.succ = t1
        if reverse_subtour:
            node = t4
            while node.pred != t4:
                temp = node.pred
                node.pred = node.succ
                node.succ = temp

                node = temp
            t1.pred = t1.succ
            t1.succ = t4
        if record:
            self.swap_stack.append((t1, t2, t3, t4, "swap_unfeasible"))

    def swap_node_between_t2_t3(self, t1, t4, t5, t6, record=True):
        t4_after_t1 = t1.succ == t4
        t6_after_t5 = t5.succ == t6
        reverse_subtour = t4_after_t1 != t6_after_t5

        if reverse_subtour:
            from_node = t6
            to_node = t5
            if t6_after_t5:
                from_node = t5
                to_node = t6

            while from_node != to_node:
                temp = from_node.pred
                from_node.pred = from_node.succ
                from_node.succ = temp
                from_node = temp
            temp = to_node.pred
            to_node.pred = to_node.succ
            to_node.succ = temp
        if t4_after_t1:
            t1.succ = t6
            t6.pred = t1
            t5.succ = t4
            t4.pred = t5
        else:
            t1.pred = t6
            t6.succ = t1
            t5.pred = t4
            t4.succ = t5

        if record:
            if reverse_subtour:
                self.swap_stack.append(
                    (t1, t4, t5, t6, "swap_node_between_t2_t3_reversed")
                )
            else:
                self.swap_stack.append((t1, t4, t5, t6, "swap_node_between_t2_t3"))

    def swap_double_bridge(self, t1, t2, t3, t4, t5, t6, t7, t8, record=True):
        self.swap_unfeasible(t1, t2, t3, t4, False, False)
        from_node = t4
        to_node = t1
        if t1.pred == t2:
            from_node = t1
            to_node = t4
        if not self.between(from_node, t5, to_node):
            temp = t5
            t5 = t8
            t8 = temp
            temp = t6
            t6 = t7
            t7 = temp
        if (t1.succ == t2 and t5.pred == t6) or (t1.pred == t2 and t5.succ == t6):
            temp = t5
            t5 = t6
            t6 = t5
            temp = t7
            t7 = t8
            t8 = temp
        self.swap_unfeasible(t5, t6, t7, t8, False, False)
        self.set_pos()
        if record:
            self.swap_stack.append(
                (t1, t2, t3, t4, t5, t6, t7, t8, "swap_double_bridge")
            )

    def __str__(self):
        curr_node = self.nodes[0]
        node_seq = str(curr_node.id)
        while curr_node.succ != self.nodes[0]:
            curr_node = curr_node.succ

            node_seq += f",{curr_node.id}"

        return f"({node_seq})"
