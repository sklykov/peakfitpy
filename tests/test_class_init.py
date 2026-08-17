# -*- coding: utf-8 -*-
"""
Test the implemented PeakFit1D class initialization.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in a command line.

@author: Sergei Klykov
@licence: MIT

"""
# %% Global imports
import numpy as np

from peakfitpy import PeakFit1D
from peakfitpy.utils.fitting_funcs import default_f_params


# %% Test functions
def test_class_initialization():
    """
    Test right and wrong specification of input parameters for PeakFit1D initialization.

    Returns
    -------
    None.

    """
    # Testing normal input data with various shapes of arrays or data structures
    x = [i for i in range(5)]; y = np.zeros(shape=(5, ))  # normal vectors
    PeakFit1D(x=x, y=y)
    x = np.asarray([i*0.5 for i in range(5)])[:, None]; y = np.zeros(shape=(5, ))  # x convertible, shape (5, 1)
    PeakFit1D(x=x, y=y)
    x = [i for i in range(5, 0, -1)]; y = np.zeros(shape=(5, 1))  # y convertible, x - reverse order
    PeakFit1D(x=x, y=y)
    x = np.asarray([0.8, -1.0, 0.5, 0.3])[:, None]; y = np.zeros(shape=(4, 1))  # both convertible, x - random order
    PeakFit1D(x=x, y=y)
    # Testing wrongly sized data
    try:
        x = np.zeros(shape=(5, 1)); y = np.zeros(shape=(5, 2))
        PeakFit1D(x, y)
        raise AssertionError("\nInitialization accepted X and Y with different shapes")
    except ValueError:
        pass
    try:
        x = np.zeros(shape=(5, 3)); y = np.zeros(shape=(5, ))
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
        x = np.asarray([0.8, -1.0, 0.5, 0.3, 10.0]); y = np.asarray([i*1.5 + 0.2 for i in range(4)].append(-1j))  # complex number in Y
        PeakFit1D(x, y)
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
    # Manually inserted values containing peak - basic test
    pf = PeakFit1D(x=np.asarray([10, 20, 30, 40, 50, 60]), y=1E2*np.asarray([1, 1.5, 2, 2.4, 1.7, 1.24]))
    pf.fit_function(); is_peak, xp, yp = pf.get_peak_values()
    if xp is not None and yp is not None:
        assert 34 <= xp <= 44 and yp >= 240, f"Defined peak {xp, yp} lays out expected ranges: x in [34, 44], y >= 240"
    else:
        raise AssertionError("\nPeak hasn't been found for the simple basic case")
    assert is_peak, "The peak should be fitted, not valley"
    
    # Test not implemented function what should be fitted
    x = np.asarray([(1.25*i + 2.2) for i in range(20)]); b = x.mean()
    y = np.exp(-(x - b*1.1)**6/13.0)  + 1.0 / x  # some undefined in a list of implemented functions function
    pf = PeakFit1D(x, y); pf.fit_function(); is_peak, xp, yp = pf.get_peak_values()
    if xp is not None and yp is not None:
        assert 14.5 <= xp <= 18.5 and yp >= 0.9, f"Defined peak {xp, yp} lays out expected ranges: x in [34, 44], y >= 240"
    else:
        raise AssertionError("\nPeak hasn't been found for the simple basic case")
    assert is_peak, "The peak should be fitted, not valley"
    
    # Test line fitting as fallback, baseline fitting
    x = np.asarray([1.2, 2.7]); y = np.asarray([-20, -31])
    pf = PeakFit1D(x, y); curve_fitted, peak_defined = pf.fit_function(); f_n = pf.best_fit[0].__name__
    assert curve_fitted and not peak_defined, f"Curve should be fitted: {curve_fitted}, peak not defined or False: {peak_defined}"
    assert f_n == "line_f", f"Line only should be fitted to 2 points, but fitted function: {f_n}"
    assert len(pf.all_fits) == 1, f"Only line can be fitted to only 2 points but fitted: {len(pf.all_fits)}"
    
    # Test fitting 3 points and number of suitable functions for it
    x = np.asarray([1.2, 2.7, 4.0]); y = np.asarray([-20, -32, -22]); pf = PeakFit1D(x, y); pf.fit_function()
    n_suitable_fits = len([f for f in default_f_params if len(default_f_params[f]) == 3]); f_n = pf.best_fit[0].__name__
    assert n_suitable_fits == len(pf.all_fits), f"# of fitted curves: {len(pf.all_fits)}, # of suitable curves: {n_suitable_fits}"
    assert f_n == "parabola_f", f"For 3 asymetric points problem the best fit should be parabola, instead got {f_n}"
