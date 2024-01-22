class Node:
    def __init__(self):
        self.id = -1
        self.pos = -1
        self.pred = None
        self.succ = None

    def __eq__(self, other):
        if other:
            return self.id == other.id
        else:
            return False

    def __gt__(self, other):
        return self.id > other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return f"({self.id})"

    def __repr__(self):
        return f"({self.id})"


class Node2D(Node):
    def __init__(self, x, y):
        super().__init__()

        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.id}:({self.x},{self.y})"

    def __repr__(self):
        return f"{self.id}:({self.x},{self.y})"


class NodePivot(Node):
    pass
