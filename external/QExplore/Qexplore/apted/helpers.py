"""Minimal Tree helper fallback for QExplore post-processing imports."""


class Tree:
    def __init__(self, text=""):
        self.text = text

    @classmethod
    def from_text(cls, text):
        return cls(text)

    def __str__(self):
        return str(self.text)
