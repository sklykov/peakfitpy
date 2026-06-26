# -*- coding: utf-8 -*-
"""
Main script with the class definition for peak fitting and retrieving properties.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
# %% Global imports
from collections.abc import Sequence
from contextlib import suppress
from numbers import Real
from typing import Any

# For compatibility between running configurations in Spyder and PyCharm IDEs
import matplotlib
import numpy as np
from numpy.typing import NDArray

with suppress(ImportError):
    matplotlib.use('Qt5Agg')


# %% Local (package-scoped) imports

# %% Module parameters
__docformat__ = "numpydoc"
RealSeq = Sequence[Real]  # for providing type hints accepting types like tuple[float], list[int]
nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]


# %% Main class def.
class PeakFit2D():
    """Base class for fitting a single peak on 2D data (function y = f(x))."""

    x_vals : np.ndarray; y_vals : np.ndarray

    def __init__(self, x: RealSeq | nparray, y: RealSeq | nparray):
        # Convert common sequence types to numpy arrays
        x = np.asarray(x) if isinstance(x, RealSeq) else x
        y = np.asarray(y) if isinstance(y, RealSeq) else y
        # Checking input data for consistency - expect only 1D arrays - vectors (or 2D array with single column provided)
        x_data_parced, y_data_parced = False, False
        if x.ndim == 1 and y.ndim == 1:
            self.x_vals = x.copy(); self.y_vals = y.copy(); x_data_parced, y_data_parced = True, True
        if x.ndim == 2 and x.shape[1] == 1:
            self.x_vals = x.ravel().copy(); x_data_parced = True
        if y.ndim == 2 and y.shape[1] == 1:
            self.y_vals = y.ravel().copy(); y_data_parced = True
        if not x_data_parced or not y_data_parced:
            raise ValueError("\nProvided X and/or Y data isn't 1D array (vector) or 2D convertable array (with a single column)")


# %% Define default export classes and methods used with import * statement (import * from peakfitpy)
__all__ = ['PeakFit2D']

# %% Only for development purposes, transfer it to test script
if __name__ == "__main__":
    pass
