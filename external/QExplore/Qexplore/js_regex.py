"""Fallback module for js_regex used by QExplore."""

import re


def compile(pattern, flags=0):
    return re.compile(pattern, flags)


def search(pattern, string, flags=0):
    return re.search(pattern, string, flags)


def match(pattern, string, flags=0):
    return re.match(pattern, string, flags)
