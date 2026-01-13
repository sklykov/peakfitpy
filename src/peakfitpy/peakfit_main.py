# -*- coding: utf-8 -*-
"""
Main script with the class definition for peak fitting and retrieving properties.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
# %% Global imports
import numpy as np

# For compatibility between running configurations in Spyder and PyCharm IDEs
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:  # will be thrown in the environment doesn't contain Qt-like library
    pass


# %% Local (package-scoped) imports

# %% Module parameters
__docformat__ = "numpydoc"


# %% Main class def.
class PeakFit2D():
    """Base class for fitting a single peak on 2D data (function y = f(x))."""

    x_vals : np.ndarray; y_vals : np.ndarray

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        # Checking input data for consistency - expect only 1D arrays - vectors (or 2D array with single column provided)
        if len(x_data.shape) == 1 and len(y_data.shape) == 1:
            self.x_vals = x_data.copy(); self.y_vals = y_data.copy()
        elif (len(x_data.shape) == 2 and x_data.shape[1] == 1) or (len(y_data.shape) == 2 and y_data.shape[1] == 1):
            self.x_vals = x_data.ravel().copy(); self.y_vals = y_data.ravel().copy()
        else:
            raise ValueError("\nProvided X and/or Y data isn't 1D array (vector) or 2D convertable array (with a single column)")


# %% Define default export classes and methods used with import * statement (import * from peakfitpy)
__all__ = ['PeakFit2D']

# %% Test as the main script
if __name__ == "__main__":
    pass
