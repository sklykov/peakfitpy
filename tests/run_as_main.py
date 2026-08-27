# -*- coding: utf-8 -*-
"""
Run some tests for peakfitpy library directly in this script without pytest usage.

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
import numpy as np

from peakfitpy import PeakFit1D
from peakfitpy.utils.fitting_funcs import (
    bump_f,
    constant_f,
    default_f_params,
    emg_f,
    gaussian_f,
    gaussian_leveled_f,
    generalized_gaussian_f,
    line_f,
    lorentzian_f,
    moffat_f,
    parabola_f,
    rayleigh_pdf_f,
    sinc_sq_f,
)

plot_all_curves_with_defaults = True  # for checking default parameters consistency
test_simple_case = False  # common manual test - well-defined peak
test_not_implemented_f = False  # Test not implemented function what should be still fitted
test_line = False  # edge case - 2 points 
test_exclude_line = False  # check that lines are excluded
test_noisy_parabola = False  # not transferred to the test_fitting, just checking the fit
test_smallest_points_valley = False  # ultimately, parabola fit to 3 points with a peak
test_4_points_peak = False  # test fitting of the peak consisting of 4 points
test_recover = False  # test the fitting capability of noised data
test_pure_noise = True  # initial points disturbed by AWGN with std = max - min (1.0)
test_sorting = False  # tests the sorting of input X and Y data during initialization
test_needle_spike = False  # test filtering out needle-like peak and criterion to filter it out
test_fallback_fit = False  # transferred to a test suit - fallback to the previous succesful fit
test_linear_peak_filter = False  # transferred to a test suit - additional filtering rule for peaks from curves with FWHM


# %% Only for development purposes
if __name__ == "__main__":

    pf = PeakFit1D(x=np.asarray([1, 2, 3]), y=np.asarray([0, 1, 0]))

    # Composed plot for both ranges
    if plot_all_curves_with_defaults:
        pf.plot_all_defaults()

    # Test fitting on the simple set
    if test_simple_case:
        pf2 =  PeakFit1D(x=np.asarray([10, 20, 30, 40, 50, 60]), y=1E2*np.asarray([1, 1.5, 2, 2.4, 1.7, 1.24]))
        pf2.find_best_fit(verbose=True, plot_best_fit=True)

    # Generate some complex examples and visualize the fitting
    if test_not_implemented_f:
        x = np.asarray([(1.25*i + 2.2) for i in range(20)]); b = x.mean()
        y = np.exp(-(x - b*1.1)**6/13.0)  + 1.0 / x  # some undefined in a list of implemented functions function
        pf = PeakFit1D(x, y); pf.find_best_fit(verbose=True, plot_best_fit=True)

    # Test some function + noise data => fitting
    if test_noisy_parabola:
        x = np.asarray([(0.57*i - 6.0) for i in range(22)])
        a, b, c = default_f_params[parabola_f.__name__]
        y = parabola_f(x, -a*1.64 - 0.27, b*3.0 - 0.15, c + 5.32)
        y = PeakFit1D.add_awgn(y, noise_fraction=4e-2)  # add Gaussian noise
        pf = PeakFit1D(x, y); pf.find_best_fit(verbose=True, plot_best_fit=True)
    
    # Edge case - 2 points fitting => line
    if test_line:
        x = np.asarray([1.2, 2.7]); y = np.asarray([-20, -31])
        pf = PeakFit1D(x, y); pf.find_best_fit(verbose=True, plot_best_fit=True, plot_norm_best_fit=True)
        print("# of fitted functions:", len(pf.all_fits))
    
    # Exclude line by providing polynomials
    if test_exclude_line:
        x = np.asarray([1.2, 2.7]); y = np.asarray([-20, -31])
        pf = PeakFit1D(x, y); best_fit, peak = pf.find_best_fit(exclude_funcs=PeakFit1D.polynomials)
        print(best_fit, peak)
         
    # Test that only limited amount of functions can be fitted to 3 points only
    if test_smallest_points_valley:
        x = np.asarray([1.2, 2.7, 4.0]); y = np.asarray([-20, -32, -22])
        pf = PeakFit1D(x, y); pf.find_best_fit(verbose=True, plot_best_fit=True)
    
    # 4 points fitting
    if test_4_points_peak:
        x = np.asarray([1.2, 1.5, 2.7, 4.0]); y = np.asarray([20, 26, 32, 22])
        pf = PeakFit1D(x, y); pf.find_best_fit(verbose=True, plot_best_fit=True)
    
    # Check basic recovering from adding a noise - check peak properties
    if test_recover:
        x = np.linspace(0.0, 1.0); params = default_f_params[gaussian_f.__name__]
        y = 10.0*gaussian_f(x, *params); y = PeakFit1D.add_awgn(y, noise_fraction=1e-1)
        pf = PeakFit1D(x, y); pf.find_best_fit(True, True); is_peak, xp, yp = pf.get_peak_values()
    
    # Check that disturbed by heavy noise (STD=1.0 of normalized noise) fitting doesn't produce needle-like peak
    if test_pure_noise:
        x = np.linspace(0.0, 1.0); params = default_f_params[gaussian_f.__name__]
        y = 5.0*gaussian_f(x, *params); y = PeakFit1D.add_awgn(y, noise_fraction=1.0)
        pf = PeakFit1D(x, y); pf.find_best_fit(True, True, selection_criterion="IC"); is_peak, xp, yp = pf.get_peak_values()
        if is_peak is not None:
            print("X peak within X range:", 0.0 < xp < 1.0, 
                  "\nY peak within min - max dataset values (no needle-like):", 0.85*y.min() <= yp <= 1.15*y.max())
        
    # Check that algorithm automatically sort unsorted data or inversed data
    if test_sorting:
        x = np.linspace(0.0, 1.0); a, b, k, d = default_f_params[lorentzian_f.__name__]
        y = lorentzian_f(x, a, b-0.12, k-5.5, d+0.1); y = PeakFit1D.add_awgn(y, noise_fraction=8.5e-2)
        pf = PeakFit1D(x, y); pf.find_best_fit(selection_criterion="IC"); is_peak_d, xp_d, yp_d = pf.get_peak_values()
        x = x[::-1]; y = y[::-1]  # inverse order
        pf = PeakFit1D(x, y); pf.find_best_fit(selection_criterion="IC"); is_peak_inv, xp_inv, yp_inv = pf.get_peak_values()
        print("Inversion has no effect on fit:", not is_peak_d and not is_peak_inv and np.isclose(xp_d, xp_inv) and np.isclose(yp_d, yp_inv))
        rng = np.random.default_rng()
        shuffled_indices = rng.permutation(x.shape[0])
        x = x[shuffled_indices]; y = y[shuffled_indices]
        pf = PeakFit1D(x, y); pf.find_best_fit(True, True, selection_criterion="IC"); is_peak_shf, xp_shf, yp_shf = pf.get_peak_values()
        print("Shuffling Data has no effect on fit:", not is_peak_d and not is_peak_shf and np.isclose(xp_shf, xp_inv)
              and np.isclose(yp_shf, yp_inv))
    
    # Check FWHM limitation for preventing needle spikes
    if test_needle_spike:
        rng = np.random.default_rng(33)
        x = np.linspace(start=-2.5, stop=1.5, num=41)
        y = rng.random(size=x.shape); y[y.shape[0]//2 - 6] = 15.0  # needle-like peak - works for Gaussian, Bump, Sech fitting
        pf = PeakFit1D(x, y); fit_res_np_l = pf.find_best_fit(verbose=True, plot_best_fit=True, filter_spikes=True, 
                                                              include_funcs=(bump_f, gaussian_leveled_f, line_f))
        # Make the peak not needle like, allow 4 points nearby => not "needle-like" peak
        y[y.shape[0]//2 - 8] = 8.7; y[y.shape[0]//2 - 5] = 13.0; y[y.shape[0]//2 - 4] = 9.2; y[y.shape[0]//2 - 7] = 12.4
        pf = PeakFit1D(x, y); fit_res_np_lp = pf.find_best_fit(verbose=True, plot_best_fit=True, include_funcs=(lorentzian_f, gaussian_f),
                                                               filter_spikes=True)
        # Test case transfer without plotting
        rng = np.random.default_rng(57)
        x = np.linspace(start=-2.5, stop=1.5, num=42)
        y = rng.random(size=x.shape); y[y.shape[0]//2 - 6] = 15.0  # needle-like peak - works for Gaussian, Bump, Sech fitting
        pf = PeakFit1D(x, y); fit_res_np_l = pf.find_best_fit(filter_spikes=True, 
                                                              include_funcs=(bump_f, gaussian_leveled_f, line_f, lorentzian_f))
        assert fit_res_np_l[0].function.__name__ == "line_f"
        # Make the peak not needle like, allow 4 points nearby => not "needle-like" peak
        y[y.shape[0]//2 - 8] = 8.7; y[y.shape[0]//2 - 5] = 13.0; y[y.shape[0]//2 - 4] = 9.2; y[y.shape[0]//2 - 7] = 12.4
        pf = PeakFit1D(x, y); fit_res_np_lp = pf.find_best_fit(filter_spikes=True, include_funcs=(lorentzian_f, gaussian_f, moffat_f),
                                                               selection_criterion="IC")
        assert fit_res_np_lp[0].function.__name__ == "gaussian_f"
        
    # Check fallback fitting
    if test_fallback_fit:
        x = np.linspace(start=-2.5, stop=1.5, num=3)*1e-2
        y = -0.5*np.linspace(start=-10.5, stop=1.5, num=3) + 5.7
        pf = PeakFit1D(x, y); fit_res_f = pf.find_best_fit(filter_spikes=True, include_funcs=(constant_f, line_f),
                                                           verbose=True, plot_best_fit=True)
        fit_res_f_2 = pf.find_best_fit(filter_spikes=True, include_funcs=(lorentzian_f, rayleigh_pdf_f, emg_f, sinc_sq_f), 
                                       verbose=True, plot_best_fit=True)
    
    # Check badly fitted peak filtering
    if test_linear_peak_filter:
        n_points = 5; rng = np.random.default_rng(n_points+2)
        x = np.linspace(start=-2.5, stop=1.5, num=n_points)*1e-2
        y = 5.0*np.linspace(start=1.5, stop=-1.5, num=n_points) + rng.random(size=n_points)
        pf = PeakFit1D(x, y)
        # winning below - Gaussian with the peak close to the start, if filter_line_fit is False
        pf.find_best_fit(filter_spikes=True, filter_line_fit=True, include_funcs=(lorentzian_f, gaussian_leveled_f, generalized_gaussian_f), 
                         verbose=True, plot_best_fit=True)
