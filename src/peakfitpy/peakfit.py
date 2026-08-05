# -*- coding: utf-8 -*-
"""
Main script with the class definition for peak fitting and retrieving properties.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
# %% Global imports
import warnings
from collections.abc import Sequence
from contextlib import suppress
from numbers import Real
from typing import Any

import matplotlib
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

# For compatibility between running configurations in Spyder and PyCharm IDEs
with suppress(ImportError):
    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from .utils.fitting_funcs import (
    bump_f,
    cosine_f,
    default_f_params,
    full_f_names,
    gaussian_f,
    gaussian_leveled_f,
    laplace_pdf_f,
    line_f,
    logistic_derivative_f,
    lorentzian_f,
    parabola_f,
    rayleigh_inv_pdf_f,
    rayleigh_pdf_f,
    sech_f,
    witch_agnesi_f,
)

# %% Module parameters
__docformat__ = "numpydoc"
RealSeq = Sequence[Real]  # for providing type hints accepting types like tuple[float], list[int]
nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]


# %% Main class def.
class PeakFit2D():
    """Base class for fitting a single peak on 2D data (function y = f(x)) assuming finite and real valued, non-constant X and Y ranges."""

    x_vals : np.ndarray; y_vals : np.ndarray; x_norm_01 : np.ndarray; x_norm_m11 : np.ndarray; y_norm_01 : np.ndarray
    x_min : Real; x_max : Real; x_range : Real; y_min : Real; y_max : Real; y_range : Real

    def __init__(self, x: RealSeq | nparray, y: RealSeq | nparray):
        """
        Accept X values that are: finite, Real unique numbers, and Y values that are: finite, Real numbers.
        
        X and Y values in the sense of Y = f(X) function.\n 
        
        Initialization logic automatically sort X values in ascending order along with corresponding Y values.\n
        
        For fitting, X and Y values normalized to the range [0.0, 1.0] and X values - additionally to the range [-1.0, 1.0].\n

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
            self.x_norm_01 = (self.x_vals.copy().astype(np.float64) - self.x_min) / self.x_range  # normalization to the [0.0, 1.0] range
            self.x_norm_m11 = (self.x_norm_01.copy().astype(np.float64) - 0.5)*2.0  # recalculation for a symmetric range [-1.0, 1.0]
        else:
            raise ValueError("\nDifference of max and min values of X data results to a zero range")
        # Normalize Y data to the range [0.0, 1.0]
        self.y_min = self.y_vals.min(); self.y_max = self.y_vals.max(); self.y_range = self.y_max - self.y_min
        if self.y_range != 0.0:
            self.y_norm_01 = (self.y_vals.copy().astype(np.float64) - self.y_min) / self.y_range
        else:
            self.y_norm_01 = np.zeros_like(self.y_vals)  # substitue with zeros, assuming that if min = max, only constant values provided
        # Available functions report
        self.functions = [gaussian_f, parabola_f, gaussian_leveled_f, lorentzian_f, line_f, sech_f, bump_f, witch_agnesi_f,
                          logistic_derivative_f, cosine_f, rayleigh_pdf_f, rayleigh_inv_pdf_f, laplace_pdf_f]
        self.function_names = [n.__name__ for n in self.functions]; self.function_ranges = ["0,1", "-1,1"]
        self.function_params = {key: default_f_params[key] for key in self.function_names if key in default_f_params}
    
    # %% Fitting
    def fit_best_norm(self):
        warnings.filterwarnings('ignore', message='Covariance of the parameters could not be estimated')  # ignore warnings during a search
        successful_fits = []  # store types of successfully fitted curves (function), its parameters + std
        x_r = self.function_ranges[0]  # identifier of a used range or key "0,1" 
        for function in self.functions:
            if x_r in self.function_params[function.__name__]:  # check that function is defined on the X range [0, 1]
                params = self.function_params[function.__name__][x_r]; params_len = len(params)  # number of parameters in a function
                if params_len <= self.x_norm_01.shape[0]:  # number of measurement 
                    try:
                        fitted_f_params = curve_fit(function, self.x_norm_01, self.y_norm_01, p0=params)[0]  # fitting
                        y_f = function(self.x_norm_01, *fitted_f_params)  # calculate function values using fitted parameters
                        successful_fits.append((function, fitted_f_params, np.std(self.y_norm_01 - y_f)))
                    except RuntimeError:
                        pass
    
    def fit_best_norm_m11(self):
        pass
    
    # %% Plotting
    def plot_norm(self, f_name: str='gaussian_f', x_range: str="0,1"):
        """
        Plot interactively provided function for X values in the range [0.0, 1.0] (x_range="0,1") or [-1.0, 1.0] (x_range="-1,1").
        
        List of available imported functions is available as the class attribute 'function_names'.

        Parameters
        ----------
        f_name : str, optional
            Function name. The default is 'gaussian_f'.
        x_range : str, optional
            Range for X values for plotting and selection of default parameters for a function. The default is "0,1".

        Returns
        -------
        None
        
        """
        if f_name in self.function_names and x_range in self.function_ranges and x_range in self.function_params[f_name]:
            i = self.function_names.index(f_name)
            if x_range == self.function_ranges[0]:
                x_norm = np.linspace(start=0.0, stop=1.0, num=251)
            else:
                x_norm = np.linspace(start=-1.0, stop=1.0, num=501)
            y_norm = self.functions[i](x_norm, *self.function_params[f_name][x_range])
            if not plt.isinteractive():
                plt.ion()
            full_f_name = full_f_names.get(f_name, 'Curve'); x_r = "[0.0, 1.0]" if x_range == self.function_ranges[0] else "[-1.0, 1.0]"
            plt.figure(f"{full_f_name} X={x_r}"); plt.plot(x_norm, y_norm, lw=2.75); plt.tight_layout()
        else:
            if f_name not in self.function_names:
                warnings.warn(f"\nFunction '{f_name}' not found in the list of supported functions: {self.functions}", stacklevel=2)
            if x_range not in self.function_ranges:
                warnings.warn(f"\n X_range '{x_range}' not recognized (supported: {self.function_ranges})", stacklevel=2)
    
    def plot_all_defaults(self):
        """
        Plot interactively all implemented curves for ranges [0, 1] and [-1, 1] with their default values.

        Returns
        -------
        None
        
        """
        if not plt.isinteractive():
            plt.ion()
        plt.figure("All curves with default parameters for [0, 1] range", figsize=(14.2, 9.0))
        x_norm = np.linspace(start=0.0, stop=1.0, num=251)
        for f_name in self.function_names:
            if self.function_ranges[0] in self.function_params[f_name]:
                i = self.function_names.index(f_name)
                y_norm = self.functions[i](x_norm, *self.function_params[f_name][self.function_ranges[0]])
                plt.plot(x_norm, y_norm, lw=2.75, label=full_f_names.get(f_name, 'Curve'))
        plt.legend(loc='best'); plt.tight_layout()
        plt.figure("All curves with default parameters for [-1, 1] range", figsize=(13, 8.5))
        x_norm = np.linspace(start=-1.0, stop=1.0, num=501)
        for f_name in self.function_names:
            if self.function_ranges[1] in self.function_params[f_name]:
                i = self.function_names.index(f_name)
                y_norm = self.functions[i](x_norm, *self.function_params[f_name][self.function_ranges[1]])
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
    
    def normalize_m11_x(self, x: Real | nparray) -> Real | nparray:
        """
        Normalize new (input) x values using the provided on the initialization data to the range [-1, 1].

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
        return (self.normalize_x(x) - 0.5)*2.0
    
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
    
    def denormalize_m11_x(self, x: Real | nparray) -> Real | nparray:
        """
        Return denormalized (from range [-1, 1]) x using originally provided X data range.

        Parameters
        ----------
        x : Real | nparray
            Either Real number or numpy array.

        Returns
        -------
        Real | nparray
            Denormalized input value(-s) by using of initial X range.
            
        """
        return self.denormalize_x(x*0.5 + 0.5) 
    
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


# %% Define default export classes and methods used with import * statement (import * from peakfitpy)
__all__ = ['PeakFit2D']
