# -*- coding: utf-8 -*-
"""
Test the implemented PeakFit1D class initialization with various parameters.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in a command line.

@author: Sergei Klykov
@license: MIT

"""
# %% Global imports
import numpy as np
import pytest

from peakfitpy import PeakFit1D


def test_class_initialization():
    """
    Test right and wrong specification of input parameters for PeakFit1D initialization.

    Returns
    -------
    None.

    """
    # Testing normal input data with various shapes of arrays or data structures
    rng = np.random.default_rng(25)
    x = [i for i in range(5)]; y = [i + 5 for i in range(5)]  # normal vectors
    PeakFit1D(x=x, y=y)
    x = np.asarray([i*0.5 for i in range(5)])[:, None]; y = [i for i in range(5)]  # x convertible, shape (5, 1)
    PeakFit1D(x=x, y=y)
    x = [i for i in range(5, 0, -1)]; y = rng.random(size=(5, 1))   # y convertible, x - reverse order
    PeakFit1D(x=x, y=y)
    x = np.asarray([0.8, -1.0, 0.5, 0.3])[:, None]; y = rng.random(size=(4, 1))  # both convertible, x - random order
    PeakFit1D(x=x, y=y)
    # Testing wrongly sized data
    try:
        x = rng.random(size=(6, 1)); y = rng.random(size=(6, 2))
        PeakFit1D(x, y)
        raise AssertionError("\nInitialization accepted X and Y with different shapes")
    except ValueError:
        pass
    try:
        x = rng.random(size=(7, 3)); y = rng.random(size=(7, ))
        PeakFit1D(x, y)
        raise AssertionError("\nInitialization accepted X and Y with different shapes")
    except ValueError:
        pass
    # Testing wrong input data
    try:
        x = np.asarray([0.8, -1.0, 0.5, 0.3]); y = np.asarray([i*1.5 + 0.2 for i in range(6)])  # different sized X and Y
        PeakFit1D(x, y)
        raise AssertionError("\nInitialization accepted X and Y with different shapes")
    except ValueError:
        pass
    try:
        x = np.asarray([0.8, -1.0, 0.5, 0.3, 1j-2]); y = np.asarray([i*1.5 + 0.2 for i in range(5)])  # complex number in X
        PeakFit1D(x, y)
        raise AssertionError("\nAccepted X with complex number in it")
    except ValueError:
        pass
    try:
        x = np.asarray([0.8, -1.0, 0.5, 0.3, 10.0]); y = [i*1.5 + 0.2 for i in range(4)]; y.append(-1j)  # complex number in Y
        PeakFit1D(x, y)
        raise AssertionError("\nAccepted X with complex number in it")
    except ValueError:
        pass
    
    # Conceptually better way of writing tests below - using the pattern and error type to catch automatically
    x = np.asarray([0.8, -1.0, 0.5, 0.3, np.nan]); y = [i*1.5 + 0.2 for i in range(5)]
    with pytest.raises(ValueError, match="NaN"):
        PeakFit1D(x, y)
    
    x = np.asarray([0.8, -1.0, 0.5, 0.3, 0.2]); y = [i*1.5 + 0.2 for i in range(4)]; y.append(np.inf)
    with pytest.raises(ValueError, match="infinite"):
        PeakFit1D(x, y)
        
    x = np.asarray([0.8, -1.0, 0.5, 0.3, 0.2]); y = [0.0]*5
    with pytest.raises(ValueError, match="identical"):
        PeakFit1D(x, y)
    
    # x = np.linspace(start=-200, stop=300, num=10**7+2); y = np.linspace(start=-400, stop=100, num=10**7+2)  # too much memory used
    x = [0.0, 1e-8, 2e-8, 3e-8, 4e-8, 1.0]; y = [i*1.5 + 0.2 for i in range(6)]  # should give the same result
    with pytest.raises(ValueError, match="median sampling interval of normalized X is too small"):
        PeakFit1D(x, y)
