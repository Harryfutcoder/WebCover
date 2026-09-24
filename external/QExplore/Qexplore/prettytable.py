"""Tiny PrettyTable fallback sufficient for QExplore logging paths."""


class PrettyTable:
    def __init__(self, field_names=None):
        self.field_names = field_names or []
        self.rows = []

    def add_row(self, row):
        self.rows.append(list(row))

    def __str__(self):
        rows = []
        if self.field_names:
            rows.append("\t".join(map(str, self.field_names)))
        rows.extend("\t".join(map(str, row)) for row in self.rows)
        return "\n".join(rows)
