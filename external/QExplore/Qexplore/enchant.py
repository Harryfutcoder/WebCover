"""Minimal pyenchant-compatible fallback for QExplore text filtering."""


class Dict:
    def __init__(self, language="en_US"):
        self.language = language

    def check(self, word):
        text = str(word or "")
        return bool(text) and any(ch.isalpha() for ch in text)
