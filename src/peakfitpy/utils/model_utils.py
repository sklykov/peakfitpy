# -*- coding: utf-8 -*-
"""
Symbolic definitions of functions for fitting + associated properties calculation.

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
import warnings
from collections.abc import Callable
from math import acosh, e, log, pi, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from ..fit_models import (
    bump_f,
    cubic_polynomial,
    emg_f,
    gaussian_f,
    gaussian_leveled_f,
    generalized_gaussian_f,
    laplace_pdf_f,
    logistic_derivative_f,
    lorentzian_f,
    moffat_f,
    parabola_f,
    quartic_polynomial,
    rayleigh_pdf_f,
    rayleigh_pdf_mirrored_f,
    sech_f,
    sinc_sq_f,
)
from .typing_utils import FitParams

default_f_params = {"gaussian_f": [1.0, 0.5, 0.2], "parabola_f": [-4.0, 4.0, 0.0], "cubic_polynomial": [-3.472, 1.389, 2.083, 0.0],
                    "line_f": [0.0, 0.5], "quartic_polynomial": [-5.0, 12.472, -13.828, 6.356, 0.0], "constant_f": [0.5],
                    "gaussian_leveled_f": {"peak": [1.0, 0.5, 0.15, 0.0], "valley": [-1.0, 0.5, 0.15, 1.0]},
                    "lorentzian_f": {"peak": [0.1, 0.5, pi*0.1, 0.0], "valley": [0.1, 0.5, -pi*0.1, 0.0]},
                    "sech_f": {"peak": [2.0, 0.125, 0.5, 0.0], "valley": [-2.0, 0.125, 0.5, 1.0]},
                    "bump_f": {"peak": [0.5, e, 0.5, 0.0], "valley": [0.5, -e, 0.5, 1.0]},
                    "logistic_derivative_f": {"peak": [4.0, 0.125, 0.5, 0.0], "valley": [-4.0, 0.125, 0.5, 1.0]},
                    "rayleigh_pdf_f": {"peak": [0.25, 0.41, 0.0, 0.0], "valley": [0.25, -0.41, 0.0, 1.0]},
                    "laplace_pdf_f": {"peak": [0.5, 0.15, 1.0, 0.0], "valley": [0.5, 0.15, -1.0, 1.0]},
                    "emg_f": {"peak": [0.37, 0.35, 0.10, 0.15, 0.0], "valley": [-0.37, 0.35, 0.10, 0.15, 1.0]},
                    "rayleigh_pdf_mirrored_f": {"peak": [0.25, 0.41, 1.0, 0.0], "valley": [0.25, -0.41, 1.0, 1.0]},
                    "generalized_gaussian_f": {"peak": [0.2, 3.5, 0.5, 1.0, 0.0], "valley": [0.2, 3.5, 0.5, -1.0, 1.0]},
                    "moffat_f": {"peak": [1.0, 0.5, 0.255, 2.5, 0.0], "valley": [-1.0, 0.5, 0.255, 2.5, 1.0]},
                    "sinc_sq_f": {"peak": [1.0, 0.5, 0.075, 0.0], "valley": [-1.0, 0.5, 0.075, 1.0]},
                    }

full_f_names = {"gaussian_f": "Gaussian", "parabola_f": "Parabola", "gaussian_leveled_f": "Gaussian + Const",
                "lorentzian_f": "Lorentzian", "line_f": "Line", "sech_f": "Hyperbolic Secant", "bump_f": "Bump Function",
                "logistic_derivative_f": "Derivative of Logistic", "laplace_pdf_f": "Laplace PDF",
                "rayleigh_pdf_f": "Rayleigh PDF", "rayleigh_pdf_mirrored_f": "Mirrored Rayleigh PDF",
                "cubic_polynomial": "Cubic Polynomial", "quartic_polynomial" : "Quartic Polynomial",
                "generalized_gaussian_f": "Generalized Gaussian", "moffat_f": "Moffat Profile", "sinc_sq_f": "Sinc^2 Function",
                "emg_f": "Exp. Mod. Gaussian PDF", "constant_f": "Constant Line"}

# Symmetric around the max / min functions
symmetric_f_names = ("gaussian_f", "parabola_f", "gaussian_leveled_f", "lorentzian_f", "sech_f", "bump_f",
                     "logistic_derivative_f", "laplace_pdf_f", "generalized_gaussian_f", "moffat_f", "sinc_sq_f", "constant_f")

# Generic / asymmetric functions
generic_f_names = ("rayleigh_pdf_f", "rayleigh_pdf_mirrored_f", "cubic_polynomial", "quartic_polynomial", "emg_f", "line_f")

tol = 1e-6  # ultimately is zero for the X and Y ranges laying within [0.0, 1.0]


# %% Define peak type and value
def get_peak(f: Callable, fitted_params: FitParams) -> tuple[bool, bool, float, float]:
    """
    Get information of a peak (max) or minimum value for the provided function.

    Parameters
    ----------
    f : Callable
        Callable function.
    fitted_params : FitParams (Sequence[float] | NDArray[np.floating[Any]])
        Defined best (fitted) parameters of the function.

    Returns
    -------
    bool
        Function found in the list of implemented function and could provide information about the peak.
    bool
        The function defines maximum (peak), if False - the function defines minimum.
    float
        x value related to a peak.
    float
        y value related to a peak.

    """
    is_definable = False; is_max = False; x0 = 0.0; y0 = 0.0
    if f.__name__ in default_f_params:
        is_definable = True  # by default function supposed to provide a peak value
        x_min = 0.0; x_max = 1.0  # default min / max for X range
        if f.__name__ == "parabola_f":
            a, b, c = fitted_params; is_max = a < 0.0  # parabola opens downward
            if abs(a) >= tol:
                x0 = -(0.5*b)/a  # defined from the 1st derivative
                if x_min < x0 < x_max:
                    y0 = parabola_f(x0, a, b, c)
                else:
                    is_definable = False  # peak / valley lays out of provided range
            else:
                is_definable = False  # it's not really a parabola, it's just a line
        elif f.__name__ == "gaussian_f":
            k, b, c = fitted_params; is_max = k > 0.0; x0 = b
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = gaussian_f(x0, k, b, c)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "gaussian_leveled_f":
            k, b, c, d = fitted_params; is_max = k > 0.0; x0 = b
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = gaussian_leveled_f(x0, k, b, c, d)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "lorentzian_f":
            a, b, k, d = fitted_params; is_max = a*k > 0.0; x0 = b
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = lorentzian_f(x0, a, b, k, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "line_f" or f.__name__ == "constant_f":
            is_definable = False  # line cannot reveal a peak, it's just a baseline fitting function
        elif f.__name__ == "sech_f":
            k, a, b, d = fitted_params; is_max = k > 0.0; x0 = b
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = sech_f(x0, k, a, b, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "bump_f":
            b, k, m, d = fitted_params; is_max = k > 0.0; x0 = m
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = bump_f(x0, b, k, m, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "logistic_derivative_f":
            k, a, b, d = fitted_params; is_max = k > 0.0; x0 = b
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = logistic_derivative_f(x0, k, a, b, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "rayleigh_pdf_f":
            s, k, b, d = fitted_params; is_max = k > 0.0; x0 = b + s
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = rayleigh_pdf_f(x0, s, k, b, d)
            else:
                is_definable = False  # peak / valley lays out of provided range or k == 0.0
        elif f.__name__ == "rayleigh_pdf_mirrored_f":
            s, k, b, d = fitted_params; is_max = k > 0.0; x0 = b - s
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = rayleigh_pdf_mirrored_f(x0, s, k, b, d)
            else:
                is_definable = False  # peak / valley lays out of provided range or k == 0.0
        elif f.__name__ == "laplace_pdf_f":
            m, b, k, d = fitted_params; is_max = k > 0.0; x0 = m
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = laplace_pdf_f(x0, m, b, k, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "generalized_gaussian_f":
            w, st, m, k, d = fitted_params; is_max = k > 0.0; x0 = m
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = generalized_gaussian_f(x0, w, st, m, k, d)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "moffat_f":
            k, m, w, beta, d = fitted_params; is_max = k > 0.0; x0 = m
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = moffat_f(x0, k, m, w, beta, d)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "sinc_sq_f":
            k, m, w, d = fitted_params; is_max = k > 0.0; x0 = m
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = sinc_sq_f(x0, k, m, w, d)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "emg_f":
            k, m, sigma, tau, d = fitted_params; is_max = k > 0.0
            if abs(k) > tol:
                offset = 10.0*tol  # it is used for prevent to report as extreme point just value at the bound using standard tol
                if is_max:
                    result = minimize_scalar(lambda x: -emg_f(x, *fitted_params), bounds=(0.0, 1.0), method="bounded")
                else:
                    result = minimize_scalar(lambda x: emg_f(x, *fitted_params), bounds=(0.0, 1.0), method="bounded")
                if x_min + offset < result.x < x_max - offset:  # default restricted boundaries
                    x0 = result.x; y0 = emg_f(x0, k, m, sigma, tau, d)
                else:
                    is_definable = False  # found extreme point outside or exactly on the X range bounds
            else:
                is_definable = False  # scaling coefficient is close to 0.0
        elif f.__name__ == "cubic_polynomial":
            a, b, c, d = fitted_params; discriminant_dx = b**2 - 3*a*c  # f'(x) = 0 for extreme, f'(x) = 3ax^2 + 2b*x + c
            if abs(a) >= tol and discriminant_dx > 0:  # two roots - one max, one min
                x01 = (-b + sqrt(discriminant_dx))/(3.0*a); x02 = (-b - sqrt(discriminant_dx))/(3.0*a)
                x01_in_range = 0.0 < x01 < x_max; x02_in_range = 0.0 < x02 < x_max
                ya = cubic_polynomial(0.0, a, b, c, d); yb = cubic_polynomial(x_max, a, b, c, d)
                # check if the both of extreme points in range and select only the case then the curve is unimodal - has only 1 extreme point
                if x01_in_range == x02_in_range:  # both are True or False simultaneously
                    is_definable = False
                elif x01_in_range:
                    is_max_01 = 6.0*a*x01 + 2.0*b < 0.0; y01 = cubic_polynomial(x01, a, b, c, d)
                    if (is_max_01 and y01 > ya and y01 > yb) or (not is_max_01 and y01 < ya and y01 < yb):
                        is_max = is_max_01; x0 = x01; y0 = y01
                    else:
                        is_definable = False
                elif x02_in_range:
                    is_max_02 = 6.0*a*x02 + 2.0*b < 0.0; y02 = cubic_polynomial(x02, a, b, c, d)
                    if (is_max_02 and y02 > ya and y02 > yb) or (not is_max_02 and y02 < ya and y02 < yb):
                        is_max = is_max_02; x0 = x02; y0 = y02
                    else:
                        is_definable = False
            elif abs(a) < tol:  # degenerative case - effectively, this is parabola
                is_definable, is_max, x0, y0 = get_peak(parabola_f, (b, c, d))  # call of the method with the parabola function
            else:
                is_definable = False  # either there is no max / min, or it's stationary inflection point (f(x) = x^3 it is x = 0)
        elif f.__name__ == "quartic_polynomial":
            a, b, c, d, e = fitted_params; n_digits = 9
            root_tol = 10.0**(-n_digits+3)  # looser tolerance accounting for numerical root-solving uncertainty
            if abs(a) >= tol:
                # Check special extreme point that is valid for the polynomial form (x - r)^4, where r is in [0.0, 1.0]
                x_flat = -b/(4.0*a); is_flat_extreme = False
                if x_min < x_flat < x_max:
                    f1 = 4.0*a*(x_flat**3) + 3.0*b*(x_flat**2) + 2.0*c*x_flat + d  # f'(x) == 0 for the extreme point
                    f2 = 12.0*a*(x_flat**2) + 6.0*b*x_flat +  2.0*c  # f''(x) == 0 for the extreme point
                    if abs(f1) < tol and abs(f2) < tol:
                        is_flat_extreme = True
                        if a > 0.0:
                            is_max = False; x0 = x_flat; y0 = quartic_polynomial(x_flat, a, b, c, d, e)
                        else:
                            is_max = True; x0 = x_flat; y0 = quartic_polynomial(x_flat, a, b, c, d, e)
                if not is_flat_extreme:
                    roots: NDArray[np.float64]
                    roots = np.roots([4*a, 3*b, 2*c, d])  # for f'(x) = 4*a*x^3 + 3*b*x^2 + 2*c*x + d
                    # below - keep only roots with small imaginary part, the returned roots are complex and real part withing selected x range
                    real_roots = sorted([r.real for r in roots if abs(r.imag) < root_tol and x_min < r.real < x_max])
                    # keep only unique, distinct roots
                    unique_roots = []  # empty container for collecting
                    for x_r in real_roots:
                        if not unique_roots or abs(x_r - unique_roots[-1]) > root_tol:  # add 1st element or compare with the previous one (max)
                            unique_roots.append(x_r)
                    if len(unique_roots) > 0:  # 1, 2 or 3 real, distinguishable roots
                        extreme_points = []  # define peak / valley candidates, ignore stationary inflection
                        for i, x_r in enumerate(unique_roots):
                            f2 = 12.0*a*(x_r**2) + 6.0*b*x_r + 2.0*c  # f''(x_r)
                            if f2 < -tol:  # f''(x_r) < generic tol effectively
                                extreme_points.append({i: "peak"})
                            elif f2 > tol:
                                extreme_points.append({i: "valley"})
                        # below - sort out the case of not defined extreme points or 'M' and 'W' like curves as not suitable for peaks retrieval
                        # addition: also exclude 'S' shapes, check only unimodal cases where is clear peak / valley can be defined
                        if len(extreme_points) != 1:
                            is_definable = False
                        else:
                            y_a = round(quartic_polynomial(x_min, a, b, c, d, e), n_digits)
                            y_b = round(quartic_polynomial(x_max, a, b, c, d, e), n_digits)
                            i_xr = next(iter(extreme_points[0]))  # recorded index of found extreme point as the single key from dictionary
                            xr = unique_roots[i_xr]; y_xr = round(quartic_polynomial(xr, a, b, c, d, e), n_digits)
                            if extreme_points[0][i_xr] == "peak" and y_xr > y_a and y_xr > y_b:
                                is_max = True; x0 = xr; y0 = y_xr
                            elif extreme_points[0][i_xr] == "valley" and y_xr < y_a and y_xr < y_b:
                                is_max = False; x0 = xr; y0 = y_xr
                            else:
                                is_definable = False  # local peak / valley only
                    else:
                        is_definable = False  # no extreme points found
            else:  # degenerative case - effectively, this is cubic polynomial
                is_definable, is_max, x0, y0 = get_peak(cubic_polynomial, (b, c, d, e))  # call of the method with the cubic function
        else:
            is_definable = False

    return is_definable, is_max, float(x0), float(y0)


# %% Analytical FWHM
def get_fwhm(f_name: str, w_param: float, f_params: FitParams = ()) -> float:
    """
    Calculate analytically the FWHM based on width parameter 'w_param' or full set of fitted / used values for a specific function.

    Parameters
    ----------
    f_name : str
        Function name, can be get as f.__name__ where f - function from an implemented above functions. \n
        Note that not all functions from this module has the analytically calculated FWHM, e.g. polynomials.
        List of implemented functions: gaussian_f, gaussian_leveled_f, lorentzian_f, bump_f, sech_f, laplace_pdf_f, \n
        logistic_derivative_f, rayleigh_pdf_f, rayleigh_pdf_mirrored_f, generalized_gaussian_f, moffat_f, sinc_sq_f. \n
        Note that width parameter naming depends on the used in this module parameter list and naming, \n
        e.g. for 'gaussian_f' - parameter 'c'.
    w_param : float
        Width parameter.
    f_params : FitParams (Sequence[float] | NDArray[np.floating[Any]]), optional
        Required Sequence of all parameters for functions 'generalized_gaussian_f' and 'moffat_f'. The default is ().

    Returns
    -------
    float
        Estimated Full Width at Half-Maximum (FWHM).

    """
    if f_name in default_f_params:
        if f_name == "gaussian_f" or f_name == "gaussian_leveled_f":
            return 2.0*sqrt(2.0*log(2.0))*w_param
        elif f_name == "lorentzian_f":
            return 2.0*w_param
        elif f_name == "bump_f":
            return 2.0*(sqrt(log(2.0)/(1.0 + log(2.0))))*w_param
        elif f_name == "sech_f":
            return 2.0*acosh(2.0)*w_param
        elif f_name == "logistic_derivative_f":
            return 4.0*log(1.0 + sqrt(2))*w_param
        elif f_name == "rayleigh_pdf_f" or f_name == "rayleigh_pdf_mirrored_f":
            return 1.60252*w_param
        elif f_name == "laplace_pdf_f":
            return 2.0*log(2.0)*w_param
        elif f_name == "generalized_gaussian_f":
            w, st, m, k, d = f_params
            return 2.0*w*log(2.0)**(1.0/st)
        elif f_name == "moffat_f":
            k, m, w, beta, d = f_params
            return 2.0*w*sqrt(2.0**(1.0/beta) - 1.0)
        elif f_name == "sinc_sq_f":
            return 2.78311*w_param
        else:
            __warn_mess = f"\nProvided function '{f_name}' hasn't been found between implemented FWHM functions, check the call"
            warnings.warn(__warn_mess, stacklevel=2)
            return w_param
    else:
        __warn_mess = f"\nProvided function '{f_name}' not found in a list of implemented functions"
        warnings.warn(__warn_mess, stacklevel=2)
        return w_param


funcs_with_fwhm = ("gaussian_f", "gaussian_leveled_f", "lorentzian_f", "bump_f", "sech_f", "logistic_derivative_f",  "rayleigh_pdf_f",
                   "rayleigh_pdf_mirrored_f", "laplace_pdf_f", "generalized_gaussian_f", "moffat_f", "sinc_sq_f")


def get_fwhm_generic(f_name: str, f_params: FitParams) -> float | None:
    """
    Provide more generic implementation of FWHM analytical definition.

    Parameters
    ----------
    f_name : str
        Function name (Callable f.__name__).
    f_params : FitParams (Sequence[float] | NDArray[np.floating[Any]])
        Fitted function parameters.

    Returns
    -------
    float | None
        Estimated analytical FWHM or None if function name not found in the list of implemented ones.

    """
    if f_name in funcs_with_fwhm:
        if f_name == "gaussian_f":
            k, b, c = f_params  # as from definition
            return get_fwhm(f_name, c)
        elif f_name == "gaussian_leveled_f":
            k, b, c, d = f_params
            return get_fwhm(f_name, c)
        elif f_name == "lorentzian_f":
            a, b, k, d = f_params
            return get_fwhm(f_name, a)
        elif f_name == "bump_f":
            b, k, m, d = f_params
            return get_fwhm(f_name, b)
        elif f_name == "sech_f" or f_name == "logistic_derivative_f":
            k, a, b, d = f_params
            return get_fwhm(f_name, a)
        elif f_name == "rayleigh_pdf_f" or f_name == "rayleigh_pdf_mirrored_f":
            s, k, b, d = f_params
            return get_fwhm(f_name, s)
        elif f_name == "laplace_pdf_f":
            m, b, k, d = f_params
            return get_fwhm(f_name, b)
        elif f_name == "sinc_sq_f":
            k, m, w, d = f_params
            return get_fwhm(f_name, w)
        elif f_name == "generalized_gaussian_f" or f_name == "moffat_f":
            return get_fwhm(f_name, 1.0, f_params)
        else:
            __warn_mess = f"\nProvided function '{f_name}' found in the implemented functions but doesn't have a proper wrap"
            warnings.warn(__warn_mess, stacklevel=2)
            return None
    else:
        __warn_mess = f"\nProvided function '{f_name}' not found in a list of implemented functions"
        warnings.warn(__warn_mess, stacklevel=2)
        return None


def get_width_from_fwhm(f_name: str, fwhm: float, f_params: FitParams = ()) -> float:
    """
    Get width parameter for the function based on provided FWHM.

    Parameters
    ----------
    f_name : str
        Function name, can be get as f.__name__ where f - function from an implemented above functions. \n
        Note that not all functions from this module has the analytically calculated FWHM, e.g. polynomials.
        List of implemented functions: gaussian_f, gaussian_leveled_f, lorentzian_f, bump_f, sech_f, laplace_pdf_f, \n
        logistic_derivative_f, rayleigh_pdf_f, rayleigh_pdf_mirrored_f, generalized_gaussian_f, moffat_f, sinc_sq_f. \n
        Note that width parameter naming depends on the used in this module parameter list and naming, \n
        e.g. for 'gaussian_f' - parameter 'c'.
    fwhm : float
        Full Width at Half-Maximum (FWHM).
    f_params : FitParams (Sequence[float] | NDArray[np.floating[Any]]), optional
        Required Sequence of all parameters for functions 'generalized_gaussian_f' and 'moffat_f'. The default is ().

    Returns
    -------
    float
        Width parameter for each function estimated from FWHM.

    """
    if f_name in funcs_with_fwhm:
        if f_name == "gaussian_f" or f_name == "gaussian_leveled_f":
            return fwhm / (2.0*sqrt(2.0*log(2.0)))
        elif f_name == "lorentzian_f":
            return fwhm / 2.0
        elif f_name == "bump_f":
            return fwhm / (2.0*(sqrt(log(2.0)/(1.0 + log(2.0)))))
        elif f_name == "sech_f":
            return fwhm / (2.0*acosh(2.0))
        elif f_name == "logistic_derivative_f":
            return fwhm / (4.0*log(1.0 + sqrt(2)))
        elif f_name == "rayleigh_pdf_f" or f_name == "rayleigh_pdf_mirrored_f":
            return fwhm / (1.60252)
        elif f_name == "laplace_pdf_f":
            return fwhm / (2.0*log(2.0))
        elif f_name == "generalized_gaussian_f":
            w, st, m, k, d = f_params
            return fwhm / (2.0*log(2.0)**(1.0/st))
        elif f_name == "moffat_f":
            k, m, w, beta, d = f_params
            return fwhm / (2.0*sqrt(2.0**(1.0/beta) - 1.0))
        elif f_name == "sinc_sq_f":
            return fwhm / 2.78311
        else:
            __warn_mess = f"\nProvided function '{f_name}' hasn't been found between implemented FWHM functions, check the call"
            warnings.warn(__warn_mess, stacklevel=2)
            return fwhm
    else:
        __warn_mess = f"\nProvided function '{f_name}' not found in a list of implemented functions"
        warnings.warn(__warn_mess, stacklevel=2)
        return fwhm

# %% Define fitting parameter bounds
# Restrictions on fitting parameters for curve_fit method, e.g. for Gaussian: k - not restricted, b - to the padded X range, sigma > tol,
# rules: width of function commonly within X range, k - unrestricted, b or m (central value) - in padded X range [-1.0, 2.0] or [-2.0, 2.0],
# d - function baseline in padded Y range [-1.0, 2.0]
fwhm_max = 1.005  # allow only FWHM ~= 1.0*(x_max - x_min) as the universal max width characterization parameter for parameters calculation
d_min = -1.0; d_max = 2.0; x_min = -1.0; x_max = 2.0  # universally defined from a X range [0.0, 1.0]

w_max_gaussian = fwhm_max/get_fwhm("gaussian_f", 1.0)  # Retrieve FWHM with width = 1.0
w_max_lorentzian = fwhm_max/get_fwhm("lorentzian_f", 1.0)
w_max_sech = fwhm_max/get_fwhm("sech_f", 1.0)
w_max_bump = w_max_bump = fwhm_max / get_fwhm("bump_f", 1.0)
w_max_logistic = fwhm_max/get_fwhm("logistic_derivative_f", 1.0)
w_max_rayleigh = fwhm_max/get_fwhm("rayleigh_pdf_f", 1.0)
w_max_laplace = fwhm_max/get_fwhm("laplace_pdf_f", 1.0)
w_max_gaussian_gen = fwhm_max/get_fwhm("generalized_gaussian_f", fwhm_max, [1.0, 1.0, 0.5, 1.0, 0.0])  # with st min = 1.0
w_max_moffat = fwhm_max/get_fwhm("moffat_f", fwhm_max, [1.0, 0.5, 1.0, 10.0, 0.0])   # beta maximum = 10.0
w_sinc_sq = fwhm_max/get_fwhm("sinc_sq_f", 1.0)
w_emg_g, w_emg_tau = w_max_gaussian, 1.0/log(2.0)  # recommended estimation for Gaussian and exponential decay parts

params_boundaries = {"gaussian_f": ([tol, x_min, tol], [np.inf, x_max, w_max_gaussian]),
                     "gaussian_leveled_f": {"peak": ([tol, x_min, tol, d_min], [np.inf, x_max, w_max_gaussian, d_max]),
                                            "valley": ([-np.inf, x_min, tol, d_min], [tol, x_max, w_max_gaussian, d_max])},
                     "lorentzian_f": {"peak": ([tol, x_min, tol, d_min], [w_max_lorentzian, x_max, np.inf, d_max]),
                                      "valley": ([tol, x_min, -np.inf, d_min], [w_max_lorentzian, x_max, tol, d_max])},
                     "sech_f": {"peak": ([tol, tol, x_min, d_min], [np.inf, w_max_sech, x_max, d_max]),
                                "valley": ([-np.inf, tol, x_min, d_min], [tol, w_max_sech, x_max, d_max])},
                     "bump_f": {"peak": ([tol, tol, x_min, d_min], [w_max_bump, np.inf, x_max, d_max]),
                                "valley": ([tol, -np.inf, x_min, d_min], [w_max_bump, tol, x_max, d_max])},
                     "logistic_derivative_f": {"peak": ([tol, tol, x_min, d_min], [np.inf, w_max_logistic, x_max, d_max]),
                                               "valley": ([-np.inf, tol, x_min, d_min], [tol, w_max_logistic, x_max, d_max])},
                     "rayleigh_pdf_f": {"peak": ([tol, tol, x_min, d_min], [w_max_rayleigh, np.inf, x_max, d_max]),
                                        "valley": ([tol, -np.inf, x_min, d_min], [w_max_rayleigh, tol, x_max, d_max])},
                     "rayleigh_pdf_mirrored_f": {"peak": ([tol, tol, x_min, d_min], [w_max_rayleigh, np.inf, x_max, d_max]),
                                                 "valley": ([tol, -np.inf, x_min, d_min], [w_max_rayleigh, tol, x_max, d_max])},
                     "laplace_pdf_f": {"peak": ([x_min, tol, tol, d_min], [x_max, w_max_laplace, np.inf, d_max]),
                                       "valley": ([x_min, tol, -np.inf, d_min], [x_max, w_max_laplace, tol, d_max])},
                     "generalized_gaussian_f": {"peak": ([tol, 1.0, x_min, tol, d_min], [w_max_gaussian_gen, 10.0, x_max, np.inf, d_max]),
                                                "valley": ([tol, 1.0, x_min, -np.inf, d_min], [w_max_gaussian_gen, 10.0, x_max, tol, d_max])},
                     "moffat_f": {"peak": ([tol, x_min, tol, 0.5, d_min], [np.inf, x_max, w_max_moffat, 10.0, d_max]),
                                  "valley": ([-np.inf, x_min, tol, 0.5, d_min], [tol, x_max, w_max_moffat, 10.0, d_max])},
                     "sinc_sq_f": {"peak": ([tol, 0.0, tol, d_min], [np.inf, 1.0, w_sinc_sq, d_max]),
                                   "valley": ([-np.inf, 0.0, tol, d_min], [tol, 1.0, w_sinc_sq, d_max])},
                     "emg_f": {"peak": ([tol, x_min, tol, tol, d_min], [np.inf, x_max, w_emg_g, w_emg_tau, d_max]) ,
                               "valley": ([-np.inf, x_min, tol, tol, d_min], [tol, x_max, w_emg_g, w_emg_tau, d_max])},
                     "constant_f": ([0.0], [1.0]),
                     }

# Define the index of min width parameter to correct for using the actual sampling estimation (min FWHM from data tuning)
params_w_min_index = {"gaussian_f": 2, "gaussian_leveled_f": 2, "lorentzian_f": 0, "sech_f": 1, "bump_f": 0,
                      "logistic_derivative_f": 1, "rayleigh_pdf_f": 0, "rayleigh_pdf_mirrored_f": 0, "laplace_pdf_f": 1,
                      "generalized_gaussian_f": 0, "moffat_f": 2, "sinc_sq_f": 2, "emg_f": (2, 3),
                      }

# Define initial parameters indices for adjusting them
init_params_idx = {"gaussian_leveled_f": {'w': 2, 'm': 1, 'k': 0}, "lorentzian_f": {'w': 0, 'm': 1, 'k': 2},
                   "sech_f": {'w': 1, 'm': 2, 'k': 0}, "bump_f": {'w': 0, 'm': 2, 'k': 1},
                   "logistic_derivative_f": {'w': 1, 'm': 2, 'k': 0}, "rayleigh_pdf_f": {'w': 0, 'm': 2, 'k': 1},
                   "rayleigh_pdf_mirrored_f": {'w': 0, 'm': 2, 'k': 1}, "laplace_pdf_f": {'w': 1, 'm': 0, 'k': 2},
                   "moffat_f": {'w': 2, 'm': 1, 'k': 0}, "sinc_sq_f": {'w': 2, 'm': 1, 'k': 0},
                   "generalized_gaussian_f": {'w': 0, 'm': 2, 'k': 3},
                   }
