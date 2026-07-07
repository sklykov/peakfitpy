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
        x = np.asarray(x) if isinstance(x, Sequence) else x  # Note: Runtime check cannot be done on Generic type (Sequence[Real])
        y = np.asarray(y) if isinstance(y, Sequence) else y
        # Checking input data for consistency - expect only 1D arrays - vectors (or 2D array with single column provided)
        x_data_parced, y_data_parced = False, False
        # Normal condition for both data
        if x.ndim == 1 and y.ndim == 1:
            self.x_vals = x.copy(); self.y_vals = y.copy(); x_data_parced, y_data_parced = True, True
        # Check individually inputs 
        if x.ndim == 2 and x.shape[1] == 1:
            self.x_vals = x.ravel().copy(); x_data_parced = True
        elif x.ndim == 1:
            self.x_vals = x.copy(); x_data_parced = True
        if y.ndim == 2 and y.shape[1] == 1:
            self.y_vals = y.ravel().copy(); y_data_parced = True
        elif y.ndim == 1:
            self.y_vals = y.copy(); y_data_parced = True
        # All input data checks are failed
        if not x_data_parced or not y_data_parced:
            raise ValueError("\nProvided X and/or Y data isn't 1D array (vector) or 2D convertable array (with a single column)")
        # Check data consistency - x input data ascending
        if not np.all(self.x_vals[1:] > self.x_vals[:-1]):  # check elements sequentially shifted by 1 on both ends
            if np.unique(self.x_vals).size == self.x_vals.size:
                # Check if reversing of x values are enough for making them ascending
                x_vals_reversed = self.x_vals.copy()[::-1]
                if np.all(x_vals_reversed[1:] > x_vals_reversed[:-1]):
                    self.x_vals = x_vals_reversed; self.y_vals = self.y_vals[::-1]
                else:
                    ids = np.argsort(self.x_vals, kind='stable')  # return back indices for sorting of initial array
                    self.x_vals = self.x_vals[ids]; self.y_vals = self.y_vals[ids]  # use sorted X indices for sorting both
            else:
                raise ValueError("\nProvided X data doesn't contain all unique values")
        # Normalize data for both uniform ranges [-1.0, 1.0] and [0.0, 1.0] - useful for fits
    
    # %% Static useful methods
    @staticmethod
    def is_vect_ascending(x: nparray) -> tuple[bool, nparray]:
        """
        Check and return if x vector is unique it in ascending order.

        Parameters
        ----------
        x : nparray
            1D vector of real numbers.

        Returns
        -------
        bool
            True, if x vector checked / sorted in ascending order. False if it hasn't all unique values.
        nparray
            Sorted array in ascending order if it has all unique elements or initial array.
        """
        is_ascending = np.all(x[1:] > x[:-1]); x_return = x
        if not is_ascending:
            is_unique = np.unique(x).size == x.size
            if is_unique:
                x_rev = x.copy()[::-1]
                if np.all(x_rev[1:] > x_rev[:-1]):  # simple reverse helps to make an ascending order
                    is_ascending = True; x_return = x_rev
                else:  # sorting is required
                    x_return = np.sort(x, kind='stable'); is_ascending = True
        return is_ascending, x_return


# %% Define default export classes and methods used with import * statement (import * from peakfitpy)
__all__ = ['PeakFit2D']

# %% Only for development purposes, transfer it to test script
if __name__ == "__main__":
    pass
