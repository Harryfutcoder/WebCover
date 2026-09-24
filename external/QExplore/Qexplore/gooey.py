"""Command-line compatibility shim for QExplore.

The upstream prototype imports ``Gooey`` only as a decorator around ``main``.
Our experiments run QExplore from PowerShell with explicit CLI arguments, so a
no-op decorator preserves the command-line behavior without requiring wxPython.
"""


def Gooey(*decorator_args, **decorator_kwargs):
    if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
        return decorator_args[0]

    def _decorator(fn):
        return fn

    return _decorator

