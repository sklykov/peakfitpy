# -*- coding: utf-8 -*-
"""
Test the implemented PeakFit2D class initialization.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in a command line.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np

from peakfitpy import PeakFit2D


# %% Test functions
def test_class_initialization():
    """
    Test right and wrong specification of input parameters for PeakFit2D initialization.

    Returns
    -------
    None.

    """
    # Testing normal input data with various shapes of arrays or data structures
    x = [i for i in range(5)]; y = np.zeros(shape=(5, ))  # normal vectors
    PeakFit2D(x=x, y=y)
    x = np.asarray([i*0.5 for i in range(5)])[:, None]; y = np.zeros(shape=(5, ))  # x convertible, shape (5, 1)
    PeakFit2D(x=x, y=y)
    x = [i for i in range(5, 0, -1)]; y = np.zeros(shape=(5, 1))  # y convertible, x - reverse order
    PeakFit2D(x=x, y=y)
    x = np.asarray([0.8, -1.0, 0.5, 0.3])[:, None]; y = np.zeros(shape=(4, 1))  # both convertible, x - random order
    PeakFit2D(x=x, y=y)
    # Testing wrongly sized data
    try:
        x = np.zeros(shape=(5, 1)); y = np.zeros(shape=(5, 2))
        PeakFit2D(x, y)
        raise AssertionError("\nInitialization accepted X and Y with different shapes")
    except ValueError:
        pass
    try:
        x = np.zeros(shape=(5, 3)); y = np.zeros(shape=(5, ))
        PeakFit2D(x, y)
        raise AssertionError("\nInitialization accepted X and Y with different shapes")
    except ValueError:
        pass
    # Testing wrong input data
    try:
        x = np.asarray([0.8, -1.0, 0.5, 0.3]); y = np.asarray([i*1.5 + 0.2 for i in range(6)])  # different sized X and Y
        PeakFit2D(x, y)
        raise AssertionError("\nInitialization accepted X and Y with different shapes")
    except ValueError:
        pass
    try:
        x = np.asarray([0.8, -1.0, 0.5, 0.3, 1j-2]); y = np.asarray([i*1.5 + 0.2 for i in range(5)])  # complex number in X
        PeakFit2D(x, y)
        raise AssertionError("\nAccepted X with complex number in it")
    except ValueError:
        pass
    try:
        x = np.asarray([0.8, -1.0, 0.5, 0.3, 10.0]); y = np.asarray([i*1.5 + 0.2 for i in range(4)].append(-1j))  # complex number in Y
        PeakFit2D(x, y)
        raise AssertionError("\nAccepted X with complex number in it")
    except ValueError:
        pass
        
