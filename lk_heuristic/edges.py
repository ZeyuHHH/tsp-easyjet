class Edge:
    def __init__(self, n1, n2):
        assert n1 != n2

        if n1 < n2:
            self.n1 = n1
            self.n2 = n2
        else:
            self.n1 = n2
            self.n2 = n1

    def __eq__(self, other):
        return (self.n1 == other.n1 and self.n2 == other.n2) or (
            self.n1 == other.n2 and self.n2 == other.n1
        )

    def __hash__(self):
        return hash((self.n1.id, self.n2.id))

    def __str__(self):
        return f"({self.n1},{self.n2})"

    def __repr__(self):
        return f"({self.n1},{self.n2})"
