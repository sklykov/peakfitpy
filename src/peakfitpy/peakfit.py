# -*- coding: utf-8 -*-
"""
Main script with the class definition for peak fitting and retrieving properties.

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
# %% Imports
import random
import warnings
from collections.abc import Callable, Sequence
from contextlib import suppress
from copy import deepcopy
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from .utils.fitting_funcs import (
    bump_f,
    constant_f,
    cubic_polynomial,
    default_f_params,
    emg_f,
    full_f_names,
    funcs_with_fwhm,
    gaussian_f,
    gaussian_leveled_f,
    generalized_gaussian_f,
    generic_f_names,
    get_fwhm_generic,
    get_peak,
    laplace_pdf_f,
    line_f,
    logistic_derivative_f,
    lorentzian_f,
    moffat_f,
    parabola_f,
    params_boundaries,
    params_w_min_index,
    quartic_polynomial,
    rayleigh_pdf_f,
    rayleigh_pdf_mirrored_f,
    sech_f,
    sinc_sq_f,
    symmetric_f_names,
    tol,
)
from .utils.fitting_result import Fit1DResult, PeakResult

# %% Module parameters
__docformat__ = "numpydoc"
RealScalar = int | float | np.floating[Any] | np.integer[Any]
RealSeq = Sequence[RealScalar]  # for providing type hints accepting types like tuple[float], list[int]
nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]


# %% Main class def.
class PeakFit1D():
    """
    Base class for fitting multiple curves to 1D sampled data.

    X values are expected to be finite, real, unique and non-constant. Y values are expected to be finite, real and non-constant. \n

    """

    x_vals: nparray; y_vals: nparray; x_norm: NDArray[np.float64]; y_norm: NDArray[np.float64]
    x_min: RealScalar; x_max: RealScalar; x_range: RealScalar; y_min: RealScalar; y_max: RealScalar; y_range: RealScalar
    best_fit: Fit1DResult | None; best_fit_criteria: tuple[str, ...]; peak: PeakResult | None
    best_fit_criterion: str; all_fits: list[Fit1DResult]; selected_funcs: tuple[Callable, ...]
    functions: tuple[Callable, ...] = (gaussian_f, parabola_f, gaussian_leveled_f, lorentzian_f, line_f, sech_f, bump_f,
                                       logistic_derivative_f, rayleigh_pdf_f, laplace_pdf_f, cubic_polynomial, quartic_polynomial,
                                       rayleigh_pdf_mirrored_f, generalized_gaussian_f, moffat_f, sinc_sq_f, emg_f, constant_f)
    function_names: tuple[str, ...] = tuple(f.__name__ for f in functions)  # uses internally generator expression
    polynomials: tuple[Callable, ...] = (parabola_f, cubic_polynomial, quartic_polynomial, line_f)
    polynomial_names: tuple[str, ...] = tuple(f.__name__ for f in polynomials)

    def __init__(self, x: RealSeq | nparray, y: RealSeq | nparray):
        """
        Accept X values that are: finite, Real unique numbers, and Y values that are: finite, Real numbers.

        X and Y values in the sense of Y = f(X) function.\n

        Initialization logic automatically sort X values in ascending order along with corresponding Y values.\n

        For fitting, X and Y values normalized to the range [0.0, 1.0].\n

        Parameters
        ----------
        x : RealSeq | nparray
            RealSeq = Sequence[RealScalar] type, nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]].
        y : RealSeq | nparray
            RealSeq = Sequence[RealScalar] type, nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]].

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any of requirement on the input data not met.

        """
        # Convert common sequence types to numpy arrays
        x = np.asarray(x) if isinstance(x, Sequence) else x  # Note: Runtime check cannot be done on Generic type (Sequence[RealScalar])
        y = np.asarray(y) if isinstance(y, Sequence) else y
        # Checking input data for consistency - expect only 1D arrays - vectors (or 2D array with single column provided)
        x_data_parsed, y_data_parsed = False, False
        # Normal condition for both data
        if x.ndim == 1 and y.ndim == 1:
            self.x_vals = x.copy(); self.y_vals = y.copy(); x_data_parsed, y_data_parsed = True, True
        # Check individually inputs
        if x.ndim == 2 and x.shape[1] == 1:
            self.x_vals = x.ravel().copy(); x_data_parsed = True
        elif x.ndim == 1:
            self.x_vals = x.copy(); x_data_parsed = True
        if y.ndim == 2 and y.shape[1] == 1:
            self.y_vals = y.ravel().copy(); y_data_parsed = True
        elif y.ndim == 1:
            self.y_vals = y.copy(); y_data_parsed = True
        # All input data possible transforms are failed, check resolves to false
        if not x_data_parsed or not y_data_parsed:
            raise ValueError("\nX and/or Y data isn't 1D array (vector) or 2D convertible array (with a single column)")
        # Check that data contains only finite (no infinity and no NaNs) values
        if not np.isfinite(self.x_vals).all() or not np.isfinite(self.y_vals).all():
            raise ValueError("\nX and/or Y data contains infinite or NaN values")
        # Check that data contains only real-valued numbers
        if not np.all(np.isrealobj(self.x_vals)) or not np.all(np.isrealobj(self.y_vals)):
            raise ValueError("\nX and/or Y data contains complex (not real) values")
        # Require at least 2 samples, the minimum needed for fitting a line
        if self.x_vals.shape[0] < 2 or self.y_vals.shape[0] < 2:
            raise ValueError("\nX and/or Y data should be contain >= 2 values")
        # Check data consistency - sizes of X and Y should be equal
        if self.x_vals.shape[0] != self.y_vals.shape[0]:
            raise ValueError("\nX and Y have different sizes (number of values). Expected equally sized arrays")
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
                raise ValueError("\nProvided X data doesn't contain all unique values (i.e. some or all values are identical)")
        # Normalize X data for the uniform range [0.0, 1.0] for improving numerical fit stability
        self.x_min = self.x_vals.min(); self.x_max = self.x_vals.max(); self.x_range = self.x_max - self.x_min
        if self.x_range != 0.0:
            self.x_norm = (self.x_vals.copy() - self.x_min).astype(np.float64) / self.x_range  # normalization to the [0.0, 1.0] range
            self.x_sampling = np.median(np.diff(self.x_norm))  # for making estimation for fitting width bounds
            if 1e-7 <= self.x_sampling < 1e-6:
                if self.x_norm.shape[0] > 1E6:
                    __warn_mess = ("\nEstimated median sampling interval of normalized X is below 1E-6 " +
                                   "and number of data points is > 1E6. Consider cropping of the data.")
                else:
                    __warn_mess = ("\nEstimated median sampling interval of normalized X is below 1E-6 that is too small value. " +
                                   "Consider resampling of the data.")
                warnings.warn(__warn_mess, stacklevel=2)
            elif self.x_sampling < 1e-7:
                raise ValueError("\nEstimated median sampling interval of normalized X is too small for fitting convergence")
        # Normalize Y data to the range [0.0, 1.0]
        self.y_min = self.y_vals.min(); self.y_max = self.y_vals.max(); self.y_range = self.y_max - self.y_min
        if self.y_range != 0.0:
            self.y_norm = (self.y_vals.copy() - self.y_min).astype(np.float64) / self.y_range  # min-max normalization
        else:
            raise ValueError("\nProvided values are identical (constant)")
        # Initialize class instance variables
        self.function_params = {name: default_f_params[name].copy() for name in self.function_names}  # fail if no default param-s for a f()
        self.best_fit = None; self.peak = None; self.all_fits = []; self.best_fit_criterion = ""; self.selected_funcs = ()
        # Define available best fit criteria - exclude or include "IC" based on sample size
        generator = (len(params) for params in default_f_params.values())  # Generator expression
        max_k = max(generator) + 1  # using Generator should evaluate the max iteratively
        self.best_fit_criteria = ("RMSE", "MAE", "IC") if len(self.x_norm) > max_k + 1 else ("RMSE", "MAE")

    # %% Fitting
    def find_best_fit(self, verbose: bool = False, plot_best_fit: bool = False, plot_norm_best_fit: bool = False,
                      selection_criterion: str = "RMSE", include_funcs: tuple[Callable, ...] | None = None,
                      exclude_funcs: tuple[Callable, ...] | None = None, filter_spikes: bool = False,
                      filter_line_fit: bool = False) -> tuple[Fit1DResult, PeakResult | None] | tuple[None, None]:
        """
        Fit in a loop candidate functions for X, Y normalized data and select the best fit.

        Note that fitting results stored as the dataclass 'Fit1DResult' in self.best_fit with attributes: 'function', 'params', 'pcov', \n
        'perr', 'rmse', 'mae', 'aicc'. For getting them use code snippet: 'class_instance.best_fit.function'.\n
        Meaning of attributes: 'function' - Callable function, 'params' - np.ndarray with fitted parameters, \n
        'pcov' - returned by SciPy 'curve_fit' method 'pcov', 'perr' = np.sqrt(np.diag(pcov)) - estimation for each 'params' error,
        'rmse', 'mae' - calculated based on difference input Y - fitted_function(input X), 'aicc' - Corrected Akaike Information Criterion.\n
        Defined peak / valley stored in the class attribute 'peak' (or 'self.peak') with the named attributes: \n
        peak.is_defined - bool with True if the extreme point has been found / exists; peak.is_peak - bool with True if it's a peak, \n
        False - if it's valley and None if 'peak.is_defined' is False; peak.x and peak.y - coordinates of peak / valley if \n
        'peak.is_defined' is True and None otherwise. \n
        Note that params, pcov, perr, and PeakResult.x/y refer to the normalized fitting coordinates. \n
        Use get_peak_values(original=True) to retrieve peak/valley coordinates in the original data scale. \n
        All names of supported functions available in the class attribute PeakFit1D.function_names and as Callable functions in \n
        PeakFit1D.functions. They can be used as provided Callables to either 'include_funcs', or 'exclude_funcs'. \n
        If both 'include_funcs' and 'exclude_funcs' are provided as not None, then ValueError will be thrown.

        Parameters
        ----------
        verbose : bool, optional
            Flag for verbose printing out. The default is False.
        plot_best_fit : bool, optional
            Flag for plotting found curve + peak on the originally scaled input X and Y ranges. The default is False.
        plot_norm_best_fit : bool, optional
            Flag for plotting found curve + peak on the selected normalized X and Y ranges. The default is False.
        selection_criterion : str, optional
            Criteria for selection of the best fit. Available: "RMSE", "MAE", "IC" (mix of RMSE + minimal function flexibility). \n
            "IC" stands for "Information Criteria". The default is "RMSE" (universally computable value).\n
            Note that "IC" is available only when AICc is defined for every candidate model: n > max(k) + 1, \n
            where k is the number of fitted curve parameters + 1 for the estimated residual variance (currently n >= 8).
        include_funcs : tuple[Callable, ...] | None, optional
            Tuple with Callable functions for fitting. If both 'include_funcs' and 'exclude_funcs' are None, then \n
            all supported functions are used. The default is None.
        exclude_funcs : tuple[Callable, ...] | None, optional
            Tuple with Callable functions that should be excluded from fitting. E.g., PeakFit1D.polynomials can be used. \n
            The default is None.
        filter_spikes: bool, optional
            Flag for checking fitted functions and filter out of needle-like peaks, where fewer than 3 samples lie within \n
            approximately 1.5×FWHM and whose RMSE under the peak is in ~2 times more than for the whole fit \n
            (1-2 points only contributes to a peak). The default is False.
        filter_line_fit: bool, optional
            If line_f was not included in the fitted candidates, additionally fit a line and filter peak-shaped models
            with an analytically defined FWHM whose RMSE is worse than the line fit. \n
            If line_f was already fitted, no additional filtering is performed. The default is False.

        Returns
        -------
        tuple[Fit1DResult, PeakResult | None] | tuple[None, None]
            1st dataclass (Fit1DResult) contain best fit function result, 2nd PeakResult - peak searching result. \n
            Fit1DResult's attributes (as Fit1DResult.attribute): function: Callable - fitted callable function; \n
            params: NDArray[np.floating[Any]] - fitted parameters; \n
            pcov: NDArray[np.floating[Any]] | None - store report of curve_fit method; \n
            perr: NDArray[np.floating[Any]] | None - store np.sqrt(np.diag(pcov)), all found parameters ~ +- perr; \n
            rmse: float, mae: float - calculated for normalized X and Y ranges, aicc: float | None. \n
            Note that if no Peak can be defined for fitted function, instead of PeakResult class will be returned None. \n
            PeakResult's attributes: is_defined: bool - True if peak can be defined; is_peak: bool | None - True => peak, \n
            False => valley; x: float | None - x peak in a normalized range; y: float | None - y peak in a normalized range; \n
            fwhm: float | None - for a normalized range; x_orig : float | None - in the originally scaled range; \n
            y_orig : float | None - in the originally scaled range; fwhm_orig: float | None - in the originally scaled range \n
            Note that tuple[None, None] will be returned if no best fit found and the previous fitting was unsuccessful. \n
            Otherwise, previous best fit and peak results will be returned.

        """
        previous_selected_funcs = self.selected_funcs; previous_best_criteria = self.best_fit_criteria
        self.selected_funcs = self._get_fitting_funcs(include_funcs, exclude_funcs)  # check provided parameters + get actual func-s for run
        previous_fits = deepcopy(self.all_fits); self.all_fits = []  # default values for class attributes
        previous_criteria = self.best_fit_criterion; self.best_fit_criterion = ""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Covariance of the parameters could not be estimated')  # ignore warnings in a search
            # Fitting loop
            for function in self.selected_funcs:
                f_name = function.__name__  # string form of the function name
                params = self.function_params[f_name].copy()  # used default starting fitting parameters (centered peaks)
                # get the boundaries for fitting parameters for both ranges
                params_limits = deepcopy(params_boundaries.get(f_name, None))
                params_len = len(params)  # number of parameters in a function for checking if there is enough input X, Y for fitting
                if params_len <= self.x_norm.shape[0]:  # X values should be not exceeding number of function parameters
                    try:
                        if params_limits is None:
                            if f_name in self.polynomial_names:  # polynomials are fitted without any parameters restriction
                                p = len(default_f_params[f_name]) - 1  # degree of polynomial
                                coef = Polynomial.fit(self.x_norm, self.y_norm, deg=p).convert().coef  # get coefficients 1*c + b*x^2 ...
                                coef = np.pad(coef, (0, p + 1 - coef.size))  # pad operation is necessary, since coef can be trimmed
                                fitted_f_params = coef[::-1]  # convert from c, b, a coefficients order to a, b, c
                                pcov = None; perr = None  # default value for pcov parameter
                            else:
                                # unrestrained fitting for any function that doesn't provide the limits on its parameters
                                fitted_f_params, pcov  = curve_fit(function, self.x_norm, self.y_norm, p0=params)
                                perr = np.sqrt(np.diag(pcov))
                        else:
                            # Sampling-based minimum width corresponding to approx. FWHM_min ~= 2*x_sampling for most supported profiles
                            if f_name in params_w_min_index:
                                index = params_w_min_index.get(f_name, None)
                                if isinstance(index, int):
                                    # below not exactly 2.0 coefficient for min FWHM allowed => fails for 3 points Gaussian fitting (~= max)
                                    params_limits[0][index] = 1.975*self.x_sampling*params_limits[1][index]
                                    params[index] = params_limits[0][index] if params[index] < params_limits[0][index] else params[index]
                                elif isinstance(index, tuple):  # special case of EMG distribution, get the estimations
                                    i, j = index; params_limits[0][i] = self.x_sampling; params_limits[0][j] = self.x_sampling
                                    params[i] = self.x_sampling if params[i] < self.x_sampling else params[i]
                                    params[j] = self.x_sampling if params[j] < self.x_sampling else params[j]
                            # Below - restricted on parameters fitting using 'trf' method by default
                            fitted_f_params, pcov = curve_fit(function, self.x_norm, self.y_norm, p0=params, bounds=params_limits)
                            perr = np.sqrt(np.diag(pcov))
                        y_f = function(self.x_norm, *fitted_f_params)  # calculate function values using fitted parameters
                        diff_y = self.y_norm - y_f; rmse = np.sqrt(np.mean((diff_y)**2)); mae = np.mean(np.abs(diff_y)); aicc = None
                        if len(self.best_fit_criteria) == 3:
                            aicc = self.get_information_criteria(function, rmse)  # cannot return nan since all functions are implemented
                        self.all_fits.append(Fit1DResult(function=function, params=fitted_f_params, pcov=pcov, perr=perr, rmse=rmse,
                                                         mae=mae, aicc=aicc))  # storing all fitted parameters in a dataclass
                    except RuntimeError:
                        pass  # no successful fit found
        # Filtering out the fitted functions with bad quality
        if len(self.all_fits) > 0 and filter_spikes:
            self.filter_undersampled_peaks(verbose)
        if len(self.all_fits) > 0 and filter_line_fit:
            self.filter_linear_peaks(verbose)
        # Get the best fit after fitting loop and additional spike (needle-like) filtering
        if len(self.all_fits) > 0:
            if selection_criterion in self.best_fit_criteria:
                self.best_fit_criterion = selection_criterion
            else:
                warnings.warn((f"\nInput criterion '{selection_criterion}' not found among allowed {self.best_fit_criteria}."
                               + "Fallback to 'RMSE'"), stacklevel=2)
                self.best_fit_criterion = "RMSE"
            if self.best_fit_criterion == "RMSE":
                self.all_fits.sort(key=lambda x: x.rmse)
            elif self.best_fit_criterion == "MAE":
                self.all_fits.sort(key=lambda x: x.mae)
            elif self.best_fit_criterion == "IC":
                self.all_fits.sort(key=lambda x: x.aicc if x.aicc is not None else np.inf)  # with fallback check and option for mypy
            self.best_fit = self.all_fits[0]  # best function after implemented above sorting based on the provided criteria
            peak_params = get_peak(self.best_fit.function, self.best_fit.params); best_f_name = self.best_fit.function.__name__
            if peak_params[0]:
                fwhm = float(get_fwhm_generic(best_f_name, self.best_fit.params)) if best_f_name in funcs_with_fwhm else None
                fwhm_orig = float(self.x_range)*fwhm if fwhm is not None else None
                self.peak = PeakResult(is_defined=bool(peak_params[0]), is_peak=bool(peak_params[1]), x=float(peak_params[2]),
                                       y=float(peak_params[3]), fwhm=fwhm, x_orig=float(self.denormalize_x(peak_params[2])),
                                       y_orig=float(self.denormalize_y(peak_params[3])), fwhm_orig=fwhm_orig)
            else:
                self.peak = None  # return None instead of empty data class
            if verbose:
                self.print_fit_info()
            if plot_best_fit:
                self.plot_best_curve()
            if plot_norm_best_fit:
                self.plot_best_curve(plot_norm_best_fit)
        elif len(previous_fits) > 0 and self.best_fit is not None:
            __warn_m = "\nNo curves could be fitted for the provided values. Previous fits retained."
            warnings.warn(__warn_m, stacklevel=2); self.all_fits = deepcopy(previous_fits); self.best_fit_criterion = previous_criteria
            self.best_fit_criteria = previous_best_criteria; self.selected_funcs = previous_selected_funcs
            if verbose:
                self.print_fit_info()
            if plot_best_fit:
                self.plot_best_curve()
            if plot_norm_best_fit:
                self.plot_best_curve(plot_norm_best_fit)
        else:
            __warn_m = "\nNo curves could be fitted for the provided values."; warnings.warn(__warn_m, stacklevel=2)
        if len(self.all_fits) == 0 and len(self.selected_funcs) != len(PeakFit1D.functions) and verbose:
            sel_f_names = [f.__name__ for f in self.selected_funcs]
            print("\nSelected functions for fitting that have not been fitted:", sel_f_names)
        return self.best_fit, self.peak

    def get_peak_values(self, original: bool = True) -> tuple[bool, float, float] | tuple[None, None, None]:
        """
        Return in a tuple the flag if it is peak, x, y coordinates if the peak has been defined.

        Parameters
        ----------
        original : bool, optional
            Flag for returning the x, y in the original scales if True,\n
            else - using selected normalizing scale for X and [0,1] normalization for Y. The default is True.

        Returns
        -------
        tuple[bool, float, float] | tuple[None, None, None]
            True if it is a peak and False if it is a valley + \n
            coordinates of a defined peak x, y or None for all values if fitting hasn't been done or peak cannot be defined.

        """
        if self.best_fit is not None and self.peak is not None and self.peak.is_defined:
            if original:
                return self.peak.is_peak, self.denormalize_x(self.peak.x), self.denormalize_y(self.peak.y)
            else:
                return self.peak.is_peak, self.peak.x, self.peak.y
        else:
            return None, None, None

    def _get_fitting_funcs(self, include_funcs: tuple[Callable, ...] | None = None,
                           exclude_funcs: tuple[Callable, ...] | None = None) -> tuple[Callable, ...]:
        """
        Filter out the fitting functions based on provided inputs.

        Parameters
        ----------
        include_funcs : tuple[Callable, ...] | None, optional
            Tuple with Callable functions for fitting. If both 'include_funcs' and 'exclude_funcs' are None, \n
            all supported functions are used. The default is None.
        exclude_funcs : tuple[Callable, ...] | None, optional
            Tuple with Callable functions that should be excluded from fitting. The default is None.

        Returns
        -------
        tuple[Callable, ...]
            Tuple with Callable functions for fitting.

        Raises
        ------
        ValueError
            If both include_funcs and exclude_funcs are not None.

        """
        selected_funcs: tuple[Callable, ...] = ()  # container for storing selected functions
        if include_funcs is not None and exclude_funcs is not None:
            raise ValueError("\nProviding both include_funcs and exclude_funcs as not None is too ambiguous")
        if include_funcs is not None:
            selected_funcs = tuple(f for f in include_funcs if f in self.functions)
            if not selected_funcs:
                raise ValueError("No functions found among the implemented ones or provided empty input value")
        if exclude_funcs is not None:
            selected_funcs = tuple(f for f in self.functions if f not in exclude_funcs)
            if not selected_funcs:
                raise ValueError("All functions excluded or provided empty input value")
        if not selected_funcs:
            selected_funcs = self.functions  # by default - fitting all functions
        max_k: int = max(len(default_f_params[f.__name__]) for f in selected_funcs) + 1  # find the max length between parameters
        self.best_fit_criteria = ("RMSE", "MAE", "IC") if len(self.x_norm) > max_k + 1 else ("RMSE", "MAE")
        return selected_funcs

    def filter_undersampled_peaks(self, verbose: bool = False):
        """
        Filter out badly resolved, sharp or needle-like peaks that can be fitted to the outliers in the input data.

        Parameters
        ----------
        verbose : bool, optional
            Flag for verbose printouts. The default is False.

        Returns
        -------
        None

        """
        if len(self.all_fits) > 0:
            filtered_fits = []
            for fit_res in self.all_fits:
                f_name = fit_res.function.__name__
                if f_name in funcs_with_fwhm:
                    peak_params = get_peak(fit_res.function, fit_res.params)
                    if peak_params[0]:  # peak / valley defined
                        fwhm = get_fwhm_generic(f_name, fit_res.params)
                        xa = peak_params[2] - 0.75*fwhm; xb = peak_params[2] + 0.75*fwhm  # approximate peak width by 1.5*FWHM
                        points_under_peak_mask = (self.x_norm >= xa) & (self.x_norm <= xb)
                        if np.count_nonzero(points_under_peak_mask) >= 3:  # so, there are at least 3 points within approximated peak
                            if fit_res.rmse >= tol:
                                x_filt = self.x_norm[points_under_peak_mask]; y_filt = self.y_norm[points_under_peak_mask]
                                rmse_p = np.sqrt(np.mean((fit_res.function(x_filt, *fit_res.params) - y_filt)**2))
                                ratio = round((rmse_p / fit_res.rmse), 2)
                                if ratio < 2.05:  # So, RMSE under the peak isn't worse than ...% of total RMSE
                                    filtered_fits.append(fit_res)
                                elif verbose:
                                    print((f"Function '{f_name}' filtered out from 'all_fits' because its RMSE under peak "
                                           + f"much worse than total fit RMSE (their ratio: {ratio} > 2.05)"), flush=True)
                            else:
                                filtered_fits.append(fit_res)  # fallback for the perfect fit - add it without filtering out
                    else:
                        filtered_fits.append(fit_res)  # no peak found for the function that assumes it - for fallback
                else:
                    filtered_fits.append(fit_res)
            self.all_fits = filtered_fits
            if verbose and len(self.all_fits) == 0:
                print("All fitted functions are filtered out by 'filter_undersampled_peaks'", flush=True)

    def filter_linear_peaks(self, verbose: bool = False):
        """
        Filter the fitted peaks for functions with FWHM (defined in 'funcs_with_fwhm' variable) if line function has better RMSE.

        Note that this filter would be applied if no line has been fitted before.

        Parameters
        ----------
        verbose : bool, optional
            Flag for verbose printouts. The default is False.

        Returns
        -------
        None

        """
        # Apply the auxiliary filter only if line_f was not already fitted
        if len(self.all_fits) > 0 and not any(fit.function is line_f for fit in self.all_fits):
            pf_line = PeakFit1D(x=self.x_norm, y=self.y_norm); best_line_fit, _ = pf_line.find_best_fit(include_funcs=(line_f, ))
            filtered_fits = []
            for fit_res in self.all_fits:
                f_name = fit_res.function.__name__
                if f_name in funcs_with_fwhm:
                    peak_params = get_peak(fit_res.function, fit_res.params)
                    if peak_params[0] :  # peak / valley defined
                        if best_line_fit.rmse >= fit_res.rmse:
                            filtered_fits.append(fit_res)
                        elif verbose:
                            print((f"Function '{f_name}' filtered out from 'all_fits' because its RMSE {round(fit_res.rmse, 6)}"
                                   + f" is larger than line function RMSE {round(best_line_fit.rmse, 6)}"), flush=True)
                    else:
                        filtered_fits.append(fit_res)  # no peak found for the function that assumes it - for fallback
                else:
                    filtered_fits.append(fit_res)
            self.all_fits = filtered_fits
            if verbose and len(self.all_fits) == 0:
                print("All fitted functions are filtered out by 'filter_linear_peaks'", flush=True)

    # %% Plotting
    def plot_best_curve(self, use_norm_ranges: bool = False):
        """
        Interactively plot the found best fitted curve along with the provided values.

        Parameters
        ----------
        use_norm_ranges : bool, optional
            If False, then plot shows originally scaled values, else - in X and Y in normalized ranges. The default is False.

        Returns
        -------
        None

        """
        if self.best_fit is not None and self.peak is not None:
            fig_id = random.randint(a=0, b=999); x_plot_vals = np.linspace(start=0.0, stop=1.0, num=401)
            func_n = full_f_names.get(self.best_fit.function.__name__, "")
            if use_norm_ranges:
                plt.figure(f"Best fit result - Normalized Values {fig_id}")
                plt.plot(self.x_norm, self.y_norm, "ro", ms=7, label="Input Norm. Values")

                plt.plot(x_plot_vals, self.best_fit.function(x_plot_vals, *self.best_fit.params), lw=3.0, label=f"Fitted {func_n}")
                if self.peak.is_defined:
                    pl = "Found Peak" if self.peak.is_peak else "Found Valley"
                    plt.plot(self.peak.x, self.peak.y, "o", c='#45c70c', ms=9, label=pl)
            else:
                plt.figure(f"Best fit result - Originally Scaled Values {fig_id}")
                plt.plot(self.x_vals, self.y_vals, "ro", ms=7, label="Input Raw Values"); x_raw_scaled = self.denormalize_x(x_plot_vals)
                plt.plot(x_raw_scaled, self.interpolate_y(x_raw_scaled), lw=3.0, label=f"Fitted {func_n}")
                if self.peak.is_defined:
                    pl = "Found Peak" if self.peak.is_peak else "Found Valley"
                    is_peak, xp, yp = self.get_peak_values(); plt.plot(xp, yp, "o", c='#45c70c', ms=9, label=pl)
            plt.legend(loc='best'); plt.tight_layout()

    def plot_norm(self, f_name: str='gaussian_f'):
        """
        Plot interactively provided function for X values in the range [0.0, 1.0].

        List of available imported functions is available as the class attribute 'function_names'.

        Parameters
        ----------
        f_name : str, optional
            Function name. The default is 'gaussian_f'.

        Returns
        -------
        None

        """
        if f_name in self.function_names:
            i = self.function_names.index(f_name); x_norm = np.linspace(start=0.0, stop=1.0, num=401)
            y_norm = self.functions[i](x_norm, *self.function_params[f_name])
            full_f_name = full_f_names.get(f_name, 'Curve'); x_r = "[0.0, 1.0]"
            plt.figure(f"{full_f_name} X={x_r}"); plt.plot(x_norm, y_norm, lw=2.75); plt.tight_layout()
        else:
            if f_name not in self.function_names:
                warnings.warn(f"\nFunction '{f_name}' not found in the list of supported functions: {self.functions}", stacklevel=2)

    def plot_all_defaults(self):
        """
        Plot interactively all implemented curves for range [0, 1] with their default values.

        Returns
        -------
        None

        """
        plt.figure("All symmetric around max curves with default parameters for [0, 1] range", figsize=(11.0, 7.5))
        x_norm = np.linspace(start=0.0, stop=1.0, num=401)
        for f_name in self.function_names:
            if f_name in symmetric_f_names:
                i = self.function_names.index(f_name); y_norm = self.functions[i](x_norm, *self.function_params[f_name])
                plt.plot(x_norm, y_norm, lw=2.75, label=full_f_names.get(f_name, 'Curve'))
        plt.legend(loc='best'); plt.tight_layout()
        plt.figure("All generic / asymmetric curves with default parameters for [0, 1] range", figsize=(11.0, 7.5))
        x_norm = np.linspace(start=0.0, stop=1.0, num=401)
        for f_name in self.function_names:
            if f_name in generic_f_names:
                i = self.function_names.index(f_name)
                y_norm = self.functions[i](x_norm, *self.function_params[f_name])
                plt.plot(x_norm, y_norm, lw=2.75, label=full_f_names.get(f_name, 'Curve'))
        plt.legend(loc='best'); plt.tight_layout()

    # %% Data transformers
    def normalize_x(self, x: RealScalar | nparray) -> RealScalar | nparray:
        """
        Normalize new (input) x values using the provided on the initialization data to the range [0, 1].

        Parameters
        ----------
        x : RealScalar | nparray
            Either RealScalar number or numpy array.

        Returns
        -------
        RealScalar | nparray
            Normalized data.

        Raises
        ------
        ValueError
            If provided values lay out of range of the initially used array.

        """
        x = np.asarray(x) if isinstance(x, Sequence) else x
        if isinstance(x, np.ndarray):
            if not np.all(np.isrealobj(x)):
                raise ValueError("\nX contains complex values")
            if x.min() < self.x_min or x.max() > self.x_max:
                raise ValueError("\nMin or Max element from provided x array lays out of range of initially used x array")
            if np.any(np.isnan(x)):
                raise ValueError("\nX contains NaN values")
        else:
            if not np.isrealobj(x):
                raise ValueError("\nX is complex")
            if x < self.x_min or x > self.x_max:
                raise ValueError("\nProvided element lays out of range of the initially used x array")
            if np.isnan(x):
                raise ValueError("\nX is NaN")
        return (x - self.x_min) / self.x_range

    def denormalize_x(self, x: RealScalar | nparray) -> RealScalar | nparray:
        """
        Return denormalized (from range [0, 1]) x using originally provided X data range.

        Parameters
        ----------
        x : RealScalar | nparray
            Either RealScalar number or numpy array.

        Returns
        -------
        RealScalar | nparray
            Denormalized input value(-s) by using of initial X range.

        """
        return x*self.x_range + self.x_min

    def normalize_y(self, y: RealScalar | nparray) -> RealScalar | nparray:
        """
        Normalize new (input) y values using the provided on the initialization data to the range [0, 1].

        Parameters
        ----------
        y : RealScalar | nparray
            Either RealScalar number or numpy array.

        Returns
        -------
        RealScalar | nparray
            Normalized data.

        Raises
        ------
        ValueError
            If provided values lay out of range of the initially used array.

        """
        y = np.asarray(y) if isinstance(y, Sequence) else y
        if isinstance(y, np.ndarray):
            if not np.all(np.isrealobj(y)):
                raise ValueError("\nY contains complex values")
            if y.min() < self.y_min or y.max() > self.y_max:
                raise ValueError("\nMin or Max element from provided y array lays out of range of initially used y array")
            if np.any(np.isnan(y)):
                raise ValueError("\nY contains NaN values")
        else:
            if not np.isrealobj(y):
                raise ValueError("\nY is complex")
            if y < self.y_min or y > self.y_max:
                raise ValueError("\nProvided element lays out of range of the initially used y array")
            if np.isnan(y):
                raise ValueError("\nY is NaN")
        return (y - self.y_min) / self.y_range

    def denormalize_y(self, y: RealScalar | nparray) -> RealScalar | nparray:
        """
        Return denormalized y using originally provided Y data range.

        Parameters
        ----------
        y : RealScalar | nparray
            Either RealScalar number or numpy array.

        Returns
        -------
        RealScalar | nparray
            Denormalized input value(-s) by using of initial Y range.

        """
        return y*self.y_range + self.y_min

    # %% Transform y = f(x) results
    def interpolate_y(self, x: RealScalar | nparray) -> RealScalar | nparray | None:
        """
        Get for raw input (not normalized) values the raw output from the fitted function values with same scale as original Y values.

        I.e. interpolate input values by using best fitted function.

        Parameters
        ----------
        x : RealScalar | nparray
            Values for which fitted function calculated.

        Returns
        -------
        RealScalar | nparray | None
            Output Y values from Y = f(X) where f - best fitted function.

        """
        if self.best_fit is not None:
            return self.denormalize_y(self.best_fit.function(self.normalize_x(x), *self.best_fit.params))
        else:
            __warn_m = "\nThere are no fitted function (curve) stored for calculation"; warnings.warn(__warn_m, stacklevel=2)
            return None

    # %% Utility methods
    def get_information_criteria(self, f: Callable, rmse: float, ic_type: str = "aicc") -> float:
        """
        Get Corrected Akaike Information Criterion (AICc) or Bayesian Information Criterion (BIC).

        Lower AICc and BIC values indicate a preferable balance between goodness of fit and model complexity:\n
        more fitted parameters (k) => more flexible curve fitting. \n
        AICc: n*np.log(rmse**2) + 2*k + (2*k*(k+1))/(n - k - 1), where n = X values length, k = number of fitted function parameters + 1.\n
        BIC: n*np.log(rmse**2) + k*np.log(n)

        Parameters
        ----------
        f : Callable
            Successfully fitted function.
        rmse : float
            Calculated RMSE.
        ic_type : str, optional
            The identifier for calculating AICc is "aicc" or for BIC is "bic". The default is 'aicc'.

        Returns
        -------
        float | np.nan (what is effectively also type 'float')
            Calculated float AICc or BIC if they can be defined, np.nan function not found as implemented.

        """
        if f.__name__ in default_f_params:
            n = self.x_norm.shape[0]; k = len(default_f_params[f.__name__]) + 1  # number of function params + 1
            if rmse < tol:
                rmse = 1e-6  # clamp RMSE to the smallest meaningful value used for also in the fitting_funcs.py
            # get AICc or BIC
            if ic_type == "aicc":
                if n > k + 1:
                    return n*np.log(rmse**2) + 2*k + (2*k*(k+1))/(n - k - 1)
                else:
                    return np.nan
            elif ic_type == "bic":
                return n*np.log(rmse**2) + k*np.log(n)
            else:
                raise ValueError(f"\nProvided IC type {ic_type} not implemented 'aicc' or 'bic'")
        else:
            return np.nan

    def print_fit_info(self):
        """
        Print out the information about the best fitted function and used criterion.

        Returns
        -------
        None

        """
        if self.best_fit_criterion == "RMSE":
            print("Best fit function:", full_f_names.get(self.best_fit.function.__name__, ""),
                  f"| based on {self.best_fit_criterion}:", round(self.best_fit.rmse, 6))
        elif self.best_fit_criterion == "MAE":
            print("Best fit function:", full_f_names.get(self.best_fit.function.__name__, ""),
                  f"| based on {self.best_fit_criterion}:", round(self.best_fit.mae, 6))
        elif self.best_fit_criterion == "IC":
            print("Best fit function:", full_f_names.get(self.best_fit.function.__name__, ""),
                  f"| based on {self.best_fit_criterion}:", round(self.best_fit.aicc, 3))

    # %% Static useful methods
    @staticmethod
    def is_vect_ascending(x: nparray) -> tuple[bool, nparray]:
        """
        Check and return if x vector is unique it in ascending order.

        Data type 'nparray': NDArray[np.floating] | NDArray[np.integer]

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
        is_ascending = bool(np.all(x[1:] > x[:-1])); x_return = x
        if not is_ascending:
            is_unique = np.unique(x).size == x.size
            if is_unique:
                x_rev = x.copy()[::-1]
                if np.all(x_rev[1:] > x_rev[:-1]):  # simple reverse helps to make an ascending order
                    is_ascending = True; x_return = x_rev
                else:  # sorting is required
                    x_return = np.sort(x, kind='stable'); is_ascending = True  # type: ignore
        return is_ascending, x_return

    @staticmethod
    def add_awgn(y: nparray, noise_fraction: float = 0.075, seed: int | None = None) -> nparray:
        """
        Add to the input y array Gaussian noise with zero average ("Additive White Gaussian Noise").

        Parameters
        ----------
        y : nparray
            Original array.
        noise_fraction : float, optional
            Noise standard deviation as a fraction of np.ptp(y) or max-min difference in Y.
        seed : int | None, optional
            Optional seed for making randomized addition repeatable.

        Returns
        -------
        nparray
            y + additive Gaussian noise.

        """
        if noise_fraction > 1.0 + 1e-6 or noise_fraction < 0.0:
            raise ValueError("Noise Fraction should be in a range [0.0, 1.0]")
        rng = np.random.default_rng(seed); noise_std = noise_fraction*np.ptp(y)  # np.ptp - peak to peak or max() - min() range
        return y.copy() + rng.normal(loc=0.0, scale=noise_std, size=y.shape)

    @staticmethod
    def set_interactive_pyqt_plot():
        """
        Try to set as matplotlib backend the Qt5Agg and switch interactive plotting.

        Returns
        -------
        None

        """
        with suppress(ImportError):   # For compatibility between running configurations in IDEs
            matplotlib.use('Qt5Agg')
        if not plt.isinteractive():
            plt.ion()


# %% Define default export classes and methods used with import * statement (import * from peakfitpy)
__all__ = ['PeakFit1D']
