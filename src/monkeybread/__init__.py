from importlib.metadata import version

from . import calc, plot, stat, util

__all__ = ["calc", "stat", "plot", "util"]

__version__ = version("monkeybread")
