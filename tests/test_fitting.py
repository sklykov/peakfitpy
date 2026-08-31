# -*- coding: utf-8 -*-
"""
Test the fitting for different scenarios of PeakFit1D.

@author: Sergei Klykov
@license: MIT

"""
# %% Global imports
import numpy as np
import pytest

from peakfitpy import PeakFit1D
from peakfitpy.fit_models import (
    bump_f,
    constant_f,
    emg_f,
    gaussian_f,
    gaussian_leveled_f,
    generalized_gaussian_f,
    line_f,
    lorentzian_f,
    moffat_f,
    rayleigh_pdf_f,
    sinc_sq_f,
)
from peakfitpy.utils.model_utils import default_f_params


# %% Test func-s
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
    # Manually inserted values containing peak - basic test (well-defined peak)
    pf = PeakFit1D(x=np.asarray([10, 20, 30, 40, 50, 60]), y=1E2*np.asarray([1, 1.5, 2, 2.4, 1.7, 1.24]))
    pf.find_best_fit(); is_peak, xp, yp = pf.get_peak_values()
    if xp is not None and yp is not None:
        assert 34 <= xp <= 44 and yp >= 240, f"Defined peak {xp, yp} lays out expected ranges: x in [34, 44], y >= 240"
    else:
        raise AssertionError("\nPeak hasn't been found for the simple basic case")
    assert is_peak, "The peak should be fitted, not valley"

    # Test not implemented function what should be still fitted
    x = np.asarray([(1.25*i + 2.2) for i in range(20)]); b = x.mean()
    y = np.exp(-(x - b*1.1)**6/13.0)  + 1.0 / x  # some undefined in a list of implemented functions function
    pf = PeakFit1D(x, y); pf.find_best_fit(); is_peak, xp, yp = pf.get_peak_values()
    if xp is not None and yp is not None:
        assert 14.5 <= xp <= 18.5 and yp >= 0.9, f"Defined peak {xp, yp} lays out expected ranges: x in [34, 44], y >= 240"
    else:
        raise AssertionError("\nPeak hasn't been found for the simple basic case")
    assert is_peak, "The peak should be fitted, not valley"

    # Test line fitting as fallback, baseline fitting
    x = np.asarray([1.2, 2.7]); y = np.asarray([-20, -31])
    pf = PeakFit1D(x, y); best_fit, peak = pf.find_best_fit()
    assert best_fit is not None and peak is None, "Some curve should be fitted and peak shold be not defined"
    f_n = pf.best_fit.function.__name__
    assert f_n == "line_f", f"Line only should be fitted to 2 points, but fitted function: {f_n}"
    assert len(pf.all_fits) == 2, f"Only 2 lines can be fitted to only 2 points but fitted: {len(pf.all_fits)}"

    # Test excluding lines as polynomials from fitting
    x = np.asarray([1.2, 2.7]); y = np.asarray([-20, -31])
    pf = PeakFit1D(x, y); best_fit, peak = pf.find_best_fit(exclude_funcs=PeakFit1D.polynomials)
    condition = best_fit is not None and best_fit.function.__name__ == "constant_f" and peak is None
    assert condition, "Constant Line should be fitted - not counted as polynomial"

    # Test fitting of 3 points and number of suitable functions for it
    x = np.asarray([1.2, 2.7, 4.0]); y = np.asarray([-20, -32, -22]); pf = PeakFit1D(x, y); pf.find_best_fit()
    n_suitable_fits = len([f for f in default_f_params if len(default_f_params[f]) <= y.shape[0]]); f_n = pf.best_fit.function.__name__
    assert n_suitable_fits >= len(pf.all_fits), f"# of fitted curves: {len(pf.all_fits)}, # of suitable curves: {n_suitable_fits}"
    assert f_n == "parabola_f", f"For 3 asymmetric points problem the best fit should be parabola, instead got {f_n}"

    # Test fitting of 4 points and number of suitable functions for it
    x = np.asarray([1.2, 1.5, 2.7, 4.0]); y = np.asarray([20, 26, 32, 22]); pf = PeakFit1D(x, y); pf.find_best_fit()
    n_suitable_fits = len([f for f in default_f_params if len(default_f_params[f]) <= y.shape[0]]); f_n_pos = pf.best_fit.function.__name__
    assert n_suitable_fits >= len(pf.all_fits), f"# of fitted curves: {len(pf.all_fits)}, # of suitable curves: {n_suitable_fits}"
    y = -y;  pf = PeakFit1D(x, y); pf.find_best_fit(); f_n_neg = pf.best_fit.function.__name__
    assert f_n_pos == f_n_neg, f"Functions fitted for original and -1.0*original data should be the same, instead: {f_n_pos} and {f_n_neg}"

    # Testing stability of fitting
    x = np.linspace(0.0, 1.0); params = default_f_params[gaussian_f.__name__]
    noise_fractions = [25e-3, 5e-2, 75e-3, 85e-3, 1e-1, 125e-3]; y_clean = 10.0*gaussian_f(x, *params); init_seed = 17
    for noise in noise_fractions:
        y = PeakFit1D.add_awgn(y_clean, noise_fraction=noise, seed=init_seed)
        pf = PeakFit1D(x, y); pf.find_best_fit(); is_peak, xp, yp = pf.get_peak_values()
        if is_peak is not None:
            assert is_peak and 0.4 < xp < 0.6 and 7.5 < yp < 12.5, f"\nPeak fitted not consistent: {is_peak, xp, yp}"
            init_seed += 2

    # Test that for heavily-disturbed by AWGN noise data the fitted peak / valley isn't needle-like
    x = np.linspace(0.0, 1.0); params = default_f_params[gaussian_f.__name__]
    y = 5.0*gaussian_f(x, *params); y = PeakFit1D.add_awgn(y, noise_fraction=1.0, seed=25)
    pf = PeakFit1D(x, y); pf.find_best_fit(); is_peak, xp, yp = pf.get_peak_values()
    if is_peak is not None:
        assert 0.0 < xp < 1.0 and 0.85*y.min() <= yp <= 1.15*y.max(), ("\nFitting of noisy data results in peak: {xp, yp} - what not"
                                                                       + f" in [0.0, 1.0] X and {0.85*y.min(), 1.15*y.max()} Y ranges")

    # Test robustness against inversion and shuffling of input data
    x = np.linspace(0.0, 1.0); a, b, k, d = default_f_params[lorentzian_f.__name__]
    y = lorentzian_f(x, a, b-0.12, k-5.5, d+0.1); y = PeakFit1D.add_awgn(y, noise_fraction=8.5e-2, seed=101)
    pf = PeakFit1D(x, y); pf.find_best_fit(); is_peak_d, xp_d, yp_d = pf.get_peak_values()
    x = x[::-1]; y = y[::-1]  # inverse order
    pf = PeakFit1D(x, y); pf.find_best_fit(); is_peak_inv, xp_inv, yp_inv = pf.get_peak_values()
    assert not is_peak_d and not is_peak_inv and np.isclose(xp_d, xp_inv) and np.isclose(yp_d, yp_inv), "Inversion of X and Y data failure"
    rng = np.random.default_rng(seed=250); shuffled_indices = rng.permutation(x.shape[0])
    x = x[shuffled_indices]; y = y[shuffled_indices]  # shuffling of the data
    pf = PeakFit1D(x, y); pf.find_best_fit(); is_peak_shf, xp_shf, yp_shf = pf.get_peak_values()
    assert not is_peak_d and not is_peak_shf and np.isclose(xp_shf, xp_inv) and np.isclose(yp_shf, yp_inv), "Shuffling of X and Y data failure"


def test_fitting_features():
    """
    Test features of fitting loop.

    Returns
    -------
    None

    """
    # Test needle spikes filtering and resolving criterion
    rng = np.random.default_rng(57)
    x = np.linspace(start=-2.5, stop=1.5, num=42)
    y = rng.random(size=x.shape); y[y.shape[0]//2 - 6] = 15.0  # needle-like peak - works for Gaussian, Bump, Sech fitting
    pf = PeakFit1D(x, y); fit_res_np_l = pf.find_best_fit(filter_spikes=True,
                                                          include_funcs=(bump_f, gaussian_leveled_f, line_f, lorentzian_f))
    assert fit_res_np_l[0].function.__name__ == "line_f", "Line should be fitted to the data with single outlier"
    # Make the peak not needle like, allow 4 points nearby => not "needle-like" peak
    y[y.shape[0]//2 - 8] = 8.7; y[y.shape[0]//2 - 5] = 13.0; y[y.shape[0]//2 - 4] = 9.2; y[y.shape[0]//2 - 7] = 12.4
    pf = PeakFit1D(x, y); fit_res_np_lp = pf.find_best_fit(filter_spikes=True, include_funcs=(lorentzian_f, gaussian_f, moffat_f),
                                                           selection_criterion="IC")
    assert fit_res_np_lp[0].function.__name__ == "gaussian_f", "5 points formed a peak that should be fitted as Gaussian function"


# regex magic used for filtering out only single proper warning
@pytest.mark.filterwarnings( r"ignore:\s*.*Previous fits retained\.$:UserWarning")
def test_fitting_fallback():
    """
    Test the fallback to the previous fitted curve.

    Returns
    -------
    None

    """
    x = np.linspace(start=-2.5, stop=1.5, num=3)*1e-2; y = -0.5*np.linspace(start=-10.5, stop=1.5, num=3) + 5.7  # just 3 points line
    pf = PeakFit1D(x, y); pf.find_best_fit(filter_spikes=True, include_funcs=(constant_f, line_f))
    fr, p = pf.find_best_fit(filter_spikes=True, include_funcs=(lorentzian_f, rayleigh_pdf_f, emg_f, sinc_sq_f))
    fr_n = fr.function.__name__; condition = fr is not None and p is None and fr_n == line_f.__name__ and len(pf.all_fits) > 0
    assert condition, "Expected line fitted to the data, instead: {fr.function.__name__}"


@pytest.mark.filterwarnings(r"ignore:\s*No curves could be fitted for the provided values\.$:UserWarning")
def test_linear_peak_filter():
    """
    Test filtering out of the peak fitted to the linear data using additional flag in the method.

    Returns
    -------
    None

    """
    n_points = 5; rng = np.random.default_rng(n_points+2)
    x = np.linspace(start=-2.5, stop=1.5, num=n_points)*1e-2
    y = 5.0*np.linspace(start=1.5, stop=-1.5, num=n_points) + rng.random(size=n_points)
    pf = PeakFit1D(x, y)
    # winning below - Gaussian with the peak close to the start, if filter_line_fit is False
    fr, p = pf.find_best_fit(filter_spikes=True, filter_line_fit=True, include_funcs=(lorentzian_f, gaussian_leveled_f,
                                                                                              generalized_gaussian_f))
    assert len(pf.all_fits) == 0 and fr is None and p is None, f"Expected empty container: {pf.all_fits} and both None-s: {fr}, {p}"
