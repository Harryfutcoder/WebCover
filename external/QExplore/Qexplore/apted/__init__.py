"""Minimal APTED fallback for import-time compatibility."""


class APTED:
    def __init__(self, tree1, tree2, *args, **kwargs):
        self.tree1 = tree1
        self.tree2 = tree2

    def compute_edit_distance(self):
        return 0 if str(self.tree1) == str(self.tree2) else 1
