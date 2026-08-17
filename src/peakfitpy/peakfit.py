# -*- coding: utf-8 -*-
"""
Main script with the class definition for peak fitting and retrieving properties.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
# %% Global imports
import random
import warnings
from collections.abc import Callable, Sequence
from contextlib import suppress
from copy import deepcopy
from numbers import Real
from typing import Any

import matplotlib
import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.optimize import curve_fit

# For compatibility between running configurations in Spyder and PyCharm IDEs
with suppress(ImportError):
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from .utils.fitting_funcs import (
    bump_f,
    cosine_f,
    cubic_polynomial,
    default_f_params,
    emg_f,
    full_f_names,
    gaussian_f,
    gaussian_leveled_f,
    generalized_gaussian_f,
    generic_f_names,
    get_peak,
    laplace_pdf_f,
    line_f,
    logistic_derivative_f,
    lorentzian_f,
    moffat_f,
    parabola_f,
    params_boundaries,
    quartic_polynomial,
    rayleigh_pdf_f,
    rayleigh_pdf_mirrored_f,
    sech_f,
    sinc_sq_f,
    symmetric_f_names,
    tol,
    witch_agnesi_f,
)

# %% Module parameters
__docformat__ = "numpydoc"
RealSeq = Sequence[Real]  # for providing type hints accepting types like tuple[float], list[int]
nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]


# %% Main class def.
class PeakFit1D():
    """
    Base class for fitting multiple curves on 1D sampled data (X).

    1D data in the sense of function Y = f(X) and assuming finite and real valued, non-constant X and Y values.

    """

    x_vals : np.ndarray; y_vals : np.ndarray; x_norm : np.ndarray; y_norm : np.ndarray; best_fit_criteria: str
    x_min : Real; x_max : Real; x_range : Real; y_min : Real; y_max : Real; y_range : Real; all_fits: list
    polynomials: tuple[str]; best_fit: Callable | None; peak_params: tuple | None

    def __init__(self, x: RealSeq | nparray, y: RealSeq | nparray):
        """
        Accept X values that are: finite, Real unique numbers, and Y values that are: finite, Real numbers.

        X and Y values in the sense of Y = f(X) function.\n

        Initialization logic automatically sort X values in ascending order along with corresponding Y values.\n

        For fitting, X and Y values normalized to the range [0.0, 1.0].\n

        Parameters
        ----------
        x : RealSeq | nparray
            RealSeq = Sequence[Real] type, nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]].
        y : RealSeq | nparray
            RealSeq = Sequence[Real] type, nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]].

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any of requirement on the input data not met.

        """
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
        # All input data possible tranforms are failed, check resolves to false
        if not x_data_parced or not y_data_parced:
            raise ValueError("\nX and/or Y data isn't 1D array (vector) or 2D convertable array (with a single column)")
        # Check that data contains only finite (no infinity and no NaNs) values
        if not np.isfinite(self.x_vals).all() or not np.isfinite(self.y_vals).all():
            raise ValueError("\nX and/or Y data contains infinite or NaN values")
        # Check that data contains only real-valued numbers
        if not np.all(np.isrealobj(self.x_vals)) or not np.all(np.isrealobj(self.y_vals)):
            raise ValueError("\nX and/or Y data contains complex (not real) values")
        # Check that the input arrays has minimal length of 2 - minimal required to fit line and bump function
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
                raise ValueError("\nProvided X data doesn't contain all unique values")
        # Normalize X data for both uniform ranges [-1.0, 1.0] and [0.0, 1.0] - useful for fits
        self.x_min = self.x_vals.min(); self.x_max = self.x_vals.max(); self.x_range = self.x_max - self.x_min
        if self.x_range != 0.0:
            self.x_norm = (self.x_vals.copy().astype(np.float64) - self.x_min) / self.x_range  # normalization to the [0.0, 1.0] range
        else:
            raise ValueError("\nDifference of max and min values of X data results to a zero range")
        # Normalize Y data to the range [0.0, 1.0]
        self.y_min = self.y_vals.min(); self.y_max = self.y_vals.max(); self.y_range = self.y_max - self.y_min
        if self.y_range != 0.0:
            self.y_norm = (self.y_vals.copy().astype(np.float64) - self.y_min) / self.y_range  # min-max normalization
        else:
            self.y_norm = np.zeros_like(self.y_vals)  # substitue with zeros, assuming that if min = max, only constant values provided
        # Available functions report
        self.functions = (gaussian_f, parabola_f, gaussian_leveled_f, lorentzian_f, line_f, sech_f, bump_f, witch_agnesi_f,
                          logistic_derivative_f, cosine_f, rayleigh_pdf_f, laplace_pdf_f, cubic_polynomial, quartic_polynomial,
                          rayleigh_pdf_mirrored_f, generalized_gaussian_f, moffat_f, sinc_sq_f, emg_f)
        self.function_names = [n.__name__ for n in self.functions]
        self.function_params = {key: default_f_params[key] for key in self.function_names if key in default_f_params}
        self.best_fit = None; self.peak_params = None; self.all_fits = []; self.best_fit_criteria = ""
        self.polynomials = (parabola_f.__name__, cubic_polynomial.__name__, quartic_polynomial.__name__, line_f.__name__)

    # %% Fitting
    def fit_function(self, verbose: bool = False, plot_best_fit: bool = False, plot_norm_best_fit: bool = False) -> tuple[bool, bool]:
        """
        Fit in a loop functions for X, Y normalized data.

        Note that the Callable best function is stored as self.best_fit[0]. Defined (fitted) function parameters - in self.best_fit[1].\n
        Defined peak / valley: self.peak_params[0] - bool, flag that the position of peak can be defined,\n
        self.peak_params[1] - bool, whatever it is the peak (max) or valley (min),\n
        self.peak_params[2], self.peak_params[3] - (x, y) coordinates of defined peak / valley.

        Parameters
        ----------
        verbose : bool, optional
            Flag for verbose printing out. The default is False.
        plot_best_fit : bool, optional
            Flag for plotting found curve + peak if defined. The default is False.
        plot_norm_best_fit : bool, optional
            Flag for plotting found curve + peak if defined on the selected normalized X and Y ranges.\n
            It will be plotted along with plotting best fit on the original scales (if plot_best_fit is also True).\n
            The default is False.

        Returns
        -------
        bool
            True if some predefined curve is fitted.
        bool
            True if peak / valley (max or min Y values) can be defined.

        """
        curve_fitted = False; peak_defined = False  # default return values
        previous_fits = deepcopy(self.all_fits); self.all_fits = []  # default values for class attributes
        previous_criteria = self.best_fit_criteria; self.best_fit_criteria = ""; nan_ic_calculated = False
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Covariance of the parameters could not be estimated')  # ignore warnings in a search
            # Fitting loop
            for function in self.functions:
                params = self.function_params[function.__name__]
                params_limits = params_boundaries.get(function.__name__, None)  # get the boundaries for fitting parameters for both ranges
                params_len = len(params)  # number of parameters in a function for checking if there is enough input X, Y for fitting
                if params_len <= self.x_norm.shape[0]:  # X values should be
                    try:
                        if params_limits is None:
                            if function.__name__ in self.polynomials:  # polynomials fitted also without any parameters restriction
                                p = len(default_f_params[function.__name__]) - 1  # degree of polynomial
                                # below - convert from c, b, a coefficients order to a, b, c
                                fitted_f_params = Polynomial.fit(self.x_norm, self.y_norm, deg=p).convert().coef[::-1]
                            else:
                                fitted_f_params = curve_fit(function, self.x_norm, self.y_norm, p0=params)[0]  # unrestrained fitting
                        else:
                            # below - restrained on parameters fitting
                            fitted_f_params = curve_fit(function, self.x_norm, self.y_norm, p0=params, bounds=params_limits)[0]
                        y_f = function(self.x_norm, *fitted_f_params)  # calculate function values using fitted parameters
                        rmse = np.sqrt(np.mean((self.y_norm - y_f)**2)); criteria = self.get_best_fit_criteria(function, rmse)
                        if not nan_ic_calculated:
                            nan_ic_calculated = np.isnan(criteria)
                        self.all_fits.append((function, fitted_f_params, rmse, criteria))
                    except RuntimeError:
                        pass  # no succesful fit found
        if len(self.all_fits) > 0:
            # Select criteria for best fit selection: in any NaN calculated fallback to RMSE selection
            if nan_ic_calculated:
                self.all_fits = sorted(self.all_fits, key=lambda x: x[2]); self.best_fit_criteria = "RMSE"   # sort on RMSE
            else:
                # sort on IC - Information Criteria - balanced value between smallest RMSE and lower number of required function parameters
                self.all_fits = sorted(self.all_fits, key=lambda x: x[3]); self.best_fit_criteria = "IC"
            self.best_fit = self.all_fits[0]  # best function along with parameters with minimal RMSE
            self.peak_params = get_peak(self.best_fit[0], self.best_fit[1])
            curve_fitted = True; peak_defined = self.peak_params[0]
            if verbose:
                if self.best_fit_criteria == "RMSE":
                    print("Found best fit function:", full_f_names.get(self.best_fit[0].__name__), "| based on RMSE:", 
                          round(self.best_fit[2], 4))
                else:
                    print("Found best fit function:", full_f_names.get(self.best_fit[0].__name__), "| based on IC:", round(self.best_fit[3], 3))
            if plot_best_fit:
                fig_id = random.randint(a=0, b=999)
                if not plt.isinteractive():
                    plt.ion()
                x_plot_vals = np.linspace(start=0.0, stop=1.0, num=401)
                if plot_norm_best_fit:
                    plt.figure(f"Best fit result - Normalized Values {fig_id}")
                    plt.plot(self.x_norm, self.y_norm, "ro", ms=7, label="Input Norm. Values")
                    func_n = full_f_names.get(self.best_fit[0].__name__)
                    plt.plot(x_plot_vals, self.best_fit[0](x_plot_vals, *self.best_fit[1]), lw=3.0, label=f"Fitted {func_n}")
                    if self.peak_params[0]:
                        pl = "Found Peak" if self.peak_params[1] else "Found Valley"
                        plt.plot(self.peak_params[2], self.peak_params[3], "o", c='#45c70c', ms=9, label=pl)
                    plt.legend(loc='best'); plt.tight_layout()
                plt.figure(f"Best fit result - Originally Scaled Values {fig_id}")
                plt.plot(self.x_vals, self.y_vals, "ro", ms=7, label="Input Raw Values")
                func_n = full_f_names.get(self.best_fit[0].__name__, ""); x_raw_scaled = self.denormalize_x(x_plot_vals)
                plt.plot(x_raw_scaled, self.interpolate_y(x_raw_scaled), lw=3.0, label=f"Fitted {func_n}")
                if self.peak_params[0]:
                    pl = "Found Peak" if self.peak_params[1] else "Found Valley"
                    is_peak, xp, yp = self.get_peak_values(); plt.plot(xp, yp, "o", c='#45c70c', ms=9, label=pl)
                plt.legend(loc='best'); plt.tight_layout()
        else:
            __warn_m = "\nThere are no curve fitted for the provided values. Previous fits retained"; warnings.warn(__warn_m, stacklevel=2)
            self.best_fit = None; self.peak_params = None  # store that there is no best_fit function found
            self.all_fits = deepcopy(previous_fits); self.best_fit_criteria = previous_criteria
        return curve_fitted, peak_defined

    def get_peak_values(self, original: bool = True) -> tuple[bool, float, float] | tuple[None, None, None]:
        """
        Return in a tuple x, y coordinates if the peak has been defined.

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
        if self.best_fit is not None and self.peak_params is not None and self.peak_params[0]:
            if original:
                return self.peak_params[1], self.denormalize_x(self.peak_params[2]), self.denormalize_y(self.peak_params[3])
            else:
                return self.peak_params[1], self.peak_params[2], self.peak_params[3]
        else:
            return None, None, None

    # %% Plotting
    def plot_norm(self, f_name: str='gaussian_f'):
        """
        Plot interactively provided function for X values in the range [0.0, 1.0] (x_range="0,1") or [-1.0, 1.0] (x_range="-1,1").

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
            if not plt.isinteractive():
                plt.ion()
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
        if not plt.isinteractive():
            plt.ion()
        plt.figure("All symmetric around max curves with default parameters for [0, 1] range", figsize=(11.0, 7.5))
        x_norm = np.linspace(start=0.0, stop=1.0, num=401)
        for f_name in self.function_names:
            if f_name in symmetric_f_names:
                i = self.function_names.index(f_name)
                y_norm = self.functions[i](x_norm, *self.function_params[f_name])
                plt.plot(x_norm, y_norm, lw=2.75, label=full_f_names.get(f_name, 'Curve'))
        plt.legend(loc='best'); plt.tight_layout()
        plt.figure("All generic / assymetric curves with default parameters for [0, 1] range", figsize=(11.0, 7.5))
        x_norm = np.linspace(start=0.0, stop=1.0, num=401)
        for f_name in self.function_names:
            if f_name in generic_f_names:
                i = self.function_names.index(f_name)
                y_norm = self.functions[i](x_norm, *self.function_params[f_name])
                plt.plot(x_norm, y_norm, lw=2.75, label=full_f_names.get(f_name, 'Curve'))
        plt.legend(loc='best'); plt.tight_layout()

    # %% Data transformers
    def normalize_x(self, x: Real | nparray) -> Real | nparray:
        """
        Normalize new (input) x values using the provided on the initialization data to the range [0, 1].

        Parameters
        ----------
        x : Real | nparray
            Either Real number or numpy array.

        Returns
        -------
        Real | nparray
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
        else:
            if not np.isrealobj(x):
                raise ValueError("\nX is complex")
            if x < self.x_min or x > self.x_max:
                raise ValueError("\nProvided element lays out of range of the initially used x array")
        return (x - self.x_min) / self.x_range

    def denormalize_x(self, x: Real | nparray) -> Real | nparray:
        """
        Return denormalized (from range [0, 1]) x using originally provided X data range.

        Parameters
        ----------
        x : Real | nparray
            Either Real number or numpy array.

        Returns
        -------
        Real | nparray
            Denormalized input value(-s) by using of initial X range.

        """
        return x*self.x_range + self.x_min

    def normalize_y(self, y: Real | nparray) -> Real | nparray:
        """
        Normalize new (input) y values using the provided on the initialization data to the range [0, 1].

        Parameters
        ----------
        y : Real | nparray
            Either Real number or numpy array.

        Returns
        -------
        Real | nparray
            Normalized data.

        Raises
        ------
        ValueError
            If provided values lay out of range of the initially used array.

        """
        y = np.asarray(y) if isinstance(y, Sequence) else y
        if self.y_range != 0.0:
            if isinstance(y, np.ndarray):
                if not np.all(np.isrealobj(y)):
                    raise ValueError("\nY contains complex values")
                if y.min() < self.y_min or y.max() > self.y_max:
                    raise ValueError("\nMin or Max element from provided y array lays out of range of initially used y array")
            else:
                if not np.isrealobj(y):
                    raise ValueError("\nY is complex")
                if y < self.y_min or y > self.y_max:
                    raise ValueError("\nProvided element lays out of range of the initially used y array")
            return (y - self.y_min) / self.y_range
        else:
            if isinstance(y, Real):
                return type(y)(0)  # like explicitly int(0) or float(0)
            elif isinstance(y, np.ndarray):
                return np.zeros_like(y)

    def denormalize_y(self, y: Real | nparray) -> Real | nparray:
        """
        Return denormalized y using originally provided Y data range.

        Parameters
        ----------
        y : Real | nparray
            Either Real number or numpy array.

        Returns
        -------
        Real | nparray
            Denormalized input value(-s) by using of initial Y range.

        """
        return y*self.y_range + self.y_min

    # %% Transform y = f(x) results
    def interpolate_y(self, x: Real | nparray) -> Real | nparray | None:
        """
        Get for raw input (not normalized) values the raw output from the fitted function values with same scale as original Y values.

        I.e. interpolate input values by using best fitted function.

        Parameters
        ----------
        x : Real | nparray
            Values for which fitted function calculated.

        Returns
        -------
        Real | nparray | None
            Output Y values from Y = f(X) where f - best fitted function.

        """
        if self.best_fit is not None:
            return self.denormalize_y(self.best_fit[0](self.normalize_x(x), *self.best_fit[1]))
        else:
            __warn_m = "\nThere are no fitted function (curve) stored for calculation"; warnings.warn(__warn_m, stacklevel=2)
            return None

    # %% Utility methods
    def get_best_fit_criteria(self, f: Callable, rmse: float) -> float:
        """
        Get Corrected Akaike Information Criterion (AICc) or Bayesian Information Criterion (BIC).

        Lower AICc and BIC values indicate a preferable balance between goodness of fit and model complexity:\n
        more fitted parameters (K) => more flexible curve fitting.

        Parameters
        ----------
        f : Callable
            Succesfully fitted function.
        rmse : float
            Calculated RMSE.

        Returns
        -------
        float | np.nan (what is effectively also type 'float')
            Calculated float AICc or BIC if they can be defined, np.nan if they cannot be defined.

        """
        if f.__name__ in default_f_params:
            N = self.x_norm.shape[0]; K = len(default_f_params[f.__name__]) + 1  # number of function params + 1
            if rmse < tol:
                rmse = 1e-6  # clamp RMSE to the smallest meaninful value used for also in fitting_funcs.py
            if N > K + 1:
                return N*np.log(rmse**2) + 2*K + (2*K*(K+1))/(N - K - 1)
            else:
                return N*np.log(rmse**2) + K*np.log(N)
        else:
            return np.nan

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

    @staticmethod
    def add_awgn(y: nparray, noise_fraction: float = 0.075) -> nparray:
        """
        Add to the input y array Gaussian noise with zero average ("Additive White Gaussian Noise").

        Parameters
        ----------
        y : nparray
            Original array.
        noise_fraction : float, optional
            Max STD of noise as percentage/100. The default is 0.075.

        Returns
        -------
        nparray
            y + additive Gaussian noise.

        """
        rng = np.random.default_rng(); noise_std = noise_fraction*np.ptp(y)  # np.ptp - peak to peak or max() - min() range
        return y + rng.normal(loc=0.0, scale=noise_std, size=y.shape)


# %% Define default export classes and methods used with import * statement (import * from peakfitpy)
__all__ = ['PeakFit1D']
