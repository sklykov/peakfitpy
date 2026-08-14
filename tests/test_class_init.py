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


def test_basic_fitting():
    """
    Test different fitting scenarios.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If some expected cause hasn't happen.

    """
    pf = PeakFit2D(x=np.asarray([10, 20, 30, 40, 50, 60]), y=1E2*np.asarray([1, 1.5, 2, 2.4, 1.7, 1.24]))
    pf.fit_function(); is_peak, xp, yp = pf.get_peak_values()
    if xp is not None and yp is not None:
        assert 34 <= xp <= 44 and yp >= 240, f"Defined peak {xp, yp} lays out expected ranges: x in [34, 44], y >= 240"
    else:
        raise AssertionError("\nPeak hasn't been found for the simple basic case")
    # test the same data for the fitting on -1,1 normalized X range
    pf.fit_function(x_range="-1,1"); is_peak, xp, yp = pf.get_peak_values()
    if xp is not None and yp is not None:
        assert 34 <= xp <= 44 and yp >= 240, f"Defined peak {xp, yp} lays out expected ranges: x in [34, 44], y >= 240"
    else:
        raise AssertionError("\nPeak hasn't been found for the simple basic case")
    assert is_peak, "The peak should be fitted, not valley"
    # test not implemented function what should be fitted
    x = np.asarray([(1.25*i + 2.2) for i in range(20)]); b = x.mean()
    y = np.exp(-(x - b*1.1)**6/13.0)  + 1.0 / x  # some undefined in a list of implemented functions function
    pf = PeakFit2D(x, y); pf.fit_function(); is_peak, xp, yp = pf.get_peak_values()
    if xp is not None and yp is not None:
        assert 14.5 <= xp <= 18.5 and yp >= 0.9, f"Defined peak {xp, yp} lays out expected ranges: x in [34, 44], y >= 240"
    else:
        raise AssertionError("\nPeak hasn't been found for the simple basic case")
    assert is_peak, "The peak should be fitted, not valley"
