# -*- coding: utf-8 -*-
"""
Symbolic definitions of functions for fitting.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
from collections.abc import Callable
from math import e, pi

import numpy as np
from scipy.stats import exponnorm
from scipy.optimize import minimize_scalar

default_f_params = {"gaussian_f": [1.0, 0.5, 0.2], "parabola_f": [-4.0, 4.0, 0.0], "gaussian_leveled_f": [1.0, 0.5, 0.15, 0.0],
                    "lorentzian_f": [0.1, 0.5, pi*0.1, 0.0], "line_f": [0.0, 0.5], "sech_f": [2.0, 8.0, 0.5, 0.0],
                    "bump_f": [0.5, e, 0.5, 0.0], "witch_agnesi_f": [0.1, 0.5, 2.5, 0.0],
                    "logistic_derivative_f": [4.0, 10.0, 0.5, 0.0], "cosine_f": [1.0, pi-1e-4, 0.5*pi],
                    "rayleigh_pdf_f": [0.25, 0.41, 0.0, 0.0], "laplace_pdf_f": [0.5, 0.125, 1.0, 0.0],
                    "rayleigh_pdf_mirrored_f": [0.25, 0.41, 1.0, 0.0], "cubic_polynomial": [-3.472, 1.389, 2.083, 0.0],
                    "quartic_polynomial": [-5.0, 12.472, -13.828, 6.356, 0.0], "generalized_gaussian_f": [0.2, 3.5, 0.5, 1.0, 0.0],
                    "moffat_f": [1.0, 0.5, 0.15, 2.5, 0.0], "sinc_sq_f": [1.0, 0.5, 0.075, 0.0], "emg_f": [0.37, 0.35, 0.10, 0.15, 0.0], 
                    }

full_f_names = {"gaussian_f": "Gaussian", "parabola_f": "Parabola", "gaussian_leveled_f": "Gaussian + Const",
                "lorentzian_f": "Lorentzian", "line_f": "Line", "sech_f": "Hyperbolic Secant", "bump_f": "Bump Function",
                "witch_agnesi_f": "Witch of Agnesi", "logistic_derivative_f": "Derivative of Logistic",
                "cosine_f": "Cosine", "rayleigh_pdf_f": "Rayleigh PDF", "rayleigh_pdf_mirrored_f": "Mirrored Rayleigh PDF",
                "laplace_pdf_f": "Laplace PDF", "cubic_polynomial": "Cubic Polynomial", "quartic_polynomial" : "Quartic Polynomial",
                "generalized_gaussian_f": "Generalized Gaussian", "moffat_f": "Moffat PDF", "sinc_sq_f": "Sinc^2 Function",
                "emg_f": "Exponentially Modified Gaussian PDF"}

# Symmetric around the max / min functions
symmetric_f_names = ("gaussian_f", "parabola_f", "gaussian_leveled_f", "lorentzian_f", "sech_f", "bump_f", "witch_agnesi_f",
                     "logistic_derivative_f", "cosine_f", "laplace_pdf_f", "generalized_gaussian_f", "moffat_f", "sinc_sq_f")

# Generic / assymetric functions
generic_f_names = ("rayleigh_pdf_f", "rayleigh_pdf_mirrored_f", "cubic_polynomial", "quartic_polynomial", "emg_f", "line_f")

tol = 1e-6  # ultimately is zero for the X and Y ranges laying within [0.0, 1.0] or [-1.0, 1.0]

# Restrictions on fitting parameters for curve_fit method, e.g. for Gaussian: a - not restricted, b - to the padded X range, sigma > tol
# rules: width of function commonly in [tol, +inf), k - unrestricted, b or m (central value) - in padded X range [-1.0, 2.0] or [-2.0, 2.0],
# d - function baseline in padded Y range [-1.0, 2.0]
d_min = -1.0; d_max = 2.0; x_min = -1.0; x_max = 2.0
params_boundaries = {"gaussian_f": ([-np.inf, x_min, tol], [np.inf, x_max, np.inf]),
                     "gaussian_leveled_f" : ([-np.inf, x_min, tol, d_min], [np.inf, x_max, np.inf, d_max]),
                     "lorentzian_f": ([tol, x_min, -np.inf, d_min], [np.inf, x_max, np.inf, d_max]),
                     "sech_f": ([-np.inf, tol, x_min, d_min], [np.inf, np.inf, x_max, d_max]),
                     "bump_f": ([tol, -np.inf, x_min, d_min], [np.inf, np.inf, x_max, d_max]),
                     "witch_agnesi_f": ([tol, x_min, -np.inf, d_min], [np.inf, x_max, np.inf, d_max]),
                     "logistic_derivative_f": ([-np.inf, tol, x_min, d_min], [np.inf, np.inf, x_max, d_max]),
                     "cosine_f": ([tol, tol, -pi], [np.inf, pi-tol, pi-tol]),
                     "rayleigh_pdf_f": ([tol, -np.inf, x_min, d_min], [np.inf, np.inf, x_max, d_max]),
                     "rayleigh_pdf_mirrored_f": ([tol, -np.inf, x_min, d_min], [np.inf, np.inf, x_max, d_max]),
                     "laplace_pdf_f": ([x_min, tol, -np.inf, d_min], [x_max, np.inf, np.inf, d_max]),
                     "generalized_gaussian_f": ([tol, 1.0, x_min, -np.inf, d_min], [np.inf, 10.0, x_max, np.inf, d_max]),
                     "moffat_f": ([-np.inf, x_min, tol, 0.5, d_min], [np.inf, x_max, np.inf, 10.0, d_max]),
                     "sinc_sq_f": ([-np.inf, 0.0, tol, d_min], [np.inf, 1.0, np.inf, d_max]),
                     "emg_f": ([-np.inf, x_min, tol, tol, d_min], [np.inf, x_max, np.inf, np.inf, d_max]),
                     }


# %% Function def-s
def parabola_f(X: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray | float:
    """
    Callable parabola function for fitting.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    a : float
        Coefficient on x^2.
    b : float
        Coefficient on x.
    c : float
        Coefficient on x^0.

    Returns
    -------
    np.ndarray | float
        Y = a*X^2 + b*X + c.

    """
    return a*X*X + b*X + c


def gaussian_f(X: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray | float:
    """
    Parametric Gaussian function for fitting with zero asymptotic minimal Y value.

    Equation: a*exp((-(X-b)^2)/(2*c^2)) \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    a : float
        See equation.
    b : float
        See equation. Mean value.
    c : float
        See equation. Sigma value.

    Returns
    -------
    np.ndarray
        Y = a*exp((-(X-b)^2)/2*c^2).

    """
    return a*np.exp(-(np.power(X-b, 2))/(2.0*(c**2)))


def gaussian_leveled_f(X: np.ndarray | float, a: float, b: float, c: float, d: float) -> np.ndarray | float:
    """
    Parametric Gaussian function with fitting of non-zero level (asympotic minimal Y value).

    Equation: a*exp((-(X-b)^2)/(2*c^2)) + d.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    a : float
        See equation.
    b : float
        See equation. Mean value.
    c : float
        See equation. Sigma value.
    d : float
        Constant level.

    Returns
    -------
    np.ndarray | float
        Y = a*exp((-(X-b)^2)/2*c^2) + d.

    """
    return a*np.exp(-np.power(X-b, 2)/(2.0*(c**2))) + d


def generalized_gaussian_f(X: np.ndarray | float, w: float, st: float, m: float, k: float, d: float) -> np.ndarray | float:
    """
    Parametric generalized (with arbitrary within exp) Gaussian function.

    Equation: k*exp(-|(X-m)/w|^st) + d.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    w : float
        Width of distribution.
    st : float
        Power within exp.
    m : float
        Extreme point (symmetry point).
    k : float
        Amplitude coefficient.
    d : float
        Constant level.

    Returns
    -------
    np.ndarray | float
        Y = k*exp(-|(X-m)/w|^st) + d.

    """
    z = np.abs((X-m)/w)
    return d + k*np.exp(-np.power(z, st))


def lorentzian_f(X: np.ndarray | float, a: float, b: float, k: float, d: float) -> np.ndarray | float:
    """
    Parametric Lorentzian (Cauchy PDF) function for fitting.

    Equation: ((a*k) / (pi*(a^2 + (X-b)^2))) + d \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    a : float
        See equation (gamma value).
    b : float
        See equation. Mean value (x0).
    k : float
        Scaling coefficient (detached from 'a' or gamma value).
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    np.ndarray | float
        Y = ((a*k) / (pi*(a^2 + (X-b)^2))) + d .

    """
    return ((a*k)/(pi*(a**2 + np.power(X-b, 2)))) + d


def line_f(X: np.ndarray | float, k: float, b: float)-> np.ndarray | float:
    """
    Parametric Line function for fallback fitting.

    Equation: k*X + b.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    k : float
        Linear coefficient.
    b : float
        y = f(0) value.

    Returns
    -------
    np.ndarray | float
        Y =  k*X + b.

    """
    return k*X + b


def sech_f(X: np.ndarray | float, k: float, a: float, b: float, d: float) -> np.ndarray | float:
    """
    Parametric hyperbolic secant.

    Equation: Y = k*exp(-|a*(X-b)|) / (exp(-2.0*|a*(X-b)|) + 1.0) + d \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    k : float
        Max value (scaling), for normalized values default value = 2.0 (max at X = 0.0).
    a : float
        Scaling parameter for decay (FWHM).
    b : float
        Offset on X from 0.0 of the peak value.
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    np.ndarray | float
        Y = k*exp(-|a*(X-b)|) / (exp(-2.0*|a*(X-b)|) + 1.0) + d.

    """
    z = np.abs(a*(X-b)); exp_z = np.exp(-z)  # better for numerical stability, trick is to avoid computation of exp(z), z = huge
    return ((k*exp_z)/(1.0 + exp_z**2)) + d


def bump_f(X: np.ndarray | float, b: float, k: float, m: float, d: float) -> np.ndarray | float:
    """
    Parametric bump function.

    Equation: Y = k*exp(b^2 / ((X-m)^2 - b^2)) + d where abs(X-m) < b else d \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray | float
        Function values.
    b : float
        Support width of a bump function.
    k : float
        Scaling coefficient.
    m : float
        Shift of extreme point from X = 0.0.
    d : float
        Constant level.

    Returns
    -------
    np.ndarray | float
        Y = k*exp(b^2 / ((X-m)^2 - b^2)) + d where abs(X-m) < b else d.

    """
    # trick below required to avoid using np.where() since it first evaluates both variables and returns based on condition the right one
    if np.isscalar(X):
        if abs(X-m) < b:
            return k*np.exp(b**2 / ((X-m)**2 - b**2)) + d
        else:
            return d
    else:
        Y = np.ones_like(X, dtype=float)*d  # fill with constant d values
        mask = np.abs(X-m) < b; X_valid = (X-m)[mask]  # only where X satisfy condition
        Y[mask] = k*np.exp(b**2 / (X_valid**2 - b**2)) + d  # evaluate only on the valid X values
        return Y


def witch_agnesi_f(X: np.ndarray | float, a: float, m: float, k: float, d: float) -> np.ndarray | float:
    """
    Witch Of Agnesi function.

    Equation: Y = k*(8.0*a^3 / ((X-m)^2 + 2.0*a^2)) + d \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray | float
        Function values.
    a : float
        Equation parameter, for normalized values default is a = 1/4.
    m : float
        Central shift from 0.0 of peak value in X.
    k : float
        Scaling coefficient.
    d : float
        Constant level.

    Returns
    -------
    np.ndarray | float
        Y = k*(8.0*a^3 / ((X-m)^2 + 2.0*a^2)) + d.

    """
    return k*((8.0*a**3)/((X-m)**2 + 2.0*a**2)) + d


def logistic_derivative_f(X: np.ndarray | float, k: float, a: float, b: float, d: float) -> np.ndarray | float:
    """
    Parametric derivative of logistic function.

    Equation: Y = k*(exp(-|a*(X-b)|) / (1.0 + exp(-|a*(X-b)|))^2) + d \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    k : float
        Scaling coefficient, for normalized values default is k = 4.0 for unit height.
    a : float
        Scaling of X values coefficients.
    b : float
        Shift of the peak from X=0.0
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    np.ndarray | float
        Y = k*(exp(-|a*(X-b)|) / (1.0 + exp(-|a*(X-b)|))^2) + d.

    """
    z = np.abs(a*(X-b)); exp_z = np.exp(-z)
    return k*(exp_z / np.power((1.0 + exp_z), 2)) + d


def cosine_f(X: np.ndarray | float, k: float, a: float, b: float) -> np.ndarray | float:
    """
    Parametric cosine function.

    Equation: Y = k*cos(a*X - b). Better to use for fitting on [-1.0, 1.0] interval.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    k : float
        Scaling (amplitude) coefficient.
    a : float
        Phase scaling coefficient.
    b : float
        Phase shift.

    Returns
    -------
    np.ndarray | float
        Y = k*cos(a*X - b).

    """
    return k*np.cos(a*X-b)


def rayleigh_pdf_f(X: np.ndarray | float, sigma: float, k: float, b: float, d: float) -> np.ndarray | float:
    """
    Parametric Rayleigh distribution PDF function.

    Equation: Y = (k*(X-b) / sigma^2)*exp(-(X-b)^2/2*sigma^2) + d for x >= b else d  \n
    Source: https://en.wikipedia.org/wiki/Rayleigh_distribution

    Parameters
    ----------
    X : np.ndarray | float
        Function values.
    sigma : float
        Scaling parameter.
    k : float
        Amplitude parameter.
    b : float
        Minimal X value there function is defined.
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    np.ndarray | float
        Y = (k*(X-b) / sigma^2)*exp(-(X-b)^2/2*sigma^2) + d for x >= b else d.

    """
    sigma2 = sigma**2; X_shifted = X - b
    return np.where(X_shifted >= 0.0, ((k*X_shifted)/sigma2)*np.exp(-(X_shifted**2)/(2.0*sigma2)) + d, d)


def rayleigh_pdf_mirrored_f(X: np.ndarray | float, sigma: float, k: float, b: float, d: float) -> np.ndarray | float:
    """
    Parametric mirrored Rayleigh distribution PDF function.

    Equation: Y = (k*(b-X) / sigma^2)*exp(-(b-X)^2/2*sigma^2) + d for x <= b else d  \n
    Source: https://en.wikipedia.org/wiki/Rayleigh_distribution

    Parameters
    ----------
    X : np.ndarray | float
        Function values.
    sigma : float
        Scaling parameter.
    k : float
        Amplitude parameter.
    b : float
        Maximal X value there function is defined.
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    np.ndarray | float
        Y = (k*(b-X) / sigma^2)*exp(-(b-X)^2/2*sigma^2) + d for x <= b else d .

    """
    sigma2 = sigma**2; X_shifted = b - X
    return np.where(X_shifted >= 0.0, ((k*X_shifted)/sigma2)*np.exp(-(X_shifted**2)/(2.0*sigma2)) + d, d)


def laplace_pdf_f(X: np.ndarray | float, m: float, b: float, k: float, d: float) -> np.ndarray | float:
    """
    Parametric Laplace distribution PDF function.

    Equation: Y = k*exp(-|X-m|/b) + d \n
    Source: https://en.wikipedia.org/wiki/Laplace_distribution

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    m : float
        Mean value.
    b : float
        Scaling parameter.
    k : float
        Amplitude parameter.
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    np.ndarray | float
        Y = k*exp(-|X-m|/b) + d.

    """
    return k*np.exp(-np.abs(X - m)/b) + d


def quartic_polynomial(X: np.ndarray | float, a: float, b: float, c: float, d: float, e: float) -> np.ndarray | float:
    """
    Callable quartic polynomial function for fitting.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    a : float
        Coefficient on x^4.
    b : float
        Coefficient on x^3.
    c : float
        Coefficient on x^2.
    d : float
        Coefficient on x^1.
    e : float
        Coefficient on x^0.

    Returns
    -------
    np.ndarray | float
        Y = a*X^4 + b*X^3 + c*X^2 + d*X + e.

    """
    return a*(X**4) + b*(X**3) + c*(X**2) + d*X + e


def cubic_polynomial(X: np.ndarray | float, a: float, b: float, c: float, d: float) -> np.ndarray | float:
    """
    Callable cubic polynomial function for fitting.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    a : float
        Coefficient on x^3.
    b : float
        Coefficient on x^2.
    c : float
        Coefficient on x^1.
    d : float
        Coefficient on x^0.

    Returns
    -------
    np.ndarray | float
        Y = a*X^3 + b*X^2 + c*X + d.

    """
    return a*(X**3) + b*(X**2) + c*X + d


def moffat_f(X: np.ndarray | float, k: float, m: float, w: float, beta: float, d: float) -> np.ndarray | float:
    """
    Callable Moffat function for fitting.

    Function Y = k*((1.0 + ((X-m)/w)^2)^-beta) + d. \n
    
    Reference: https://en.wikipedia.org/wiki/Moffat_distribution

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    k : float
        Scaling coefficient.
    m : float
        Symmetry center.
    w : float
        Width of distribution.
    beta : float
        Power.
    d : float
        Constant baseline addition.

    Returns
    -------
    np.ndarray | float
        Y = k*((1.0 + ((X-m)/w)^2)^-beta) + d.

    """
    z = (X - m)/w; zb = (1.0 + z**2)**(-beta)
    return k*zb + d


def sinc_sq_f(X: np.ndarray | float, k: float, m: float, w: float, d: float) -> np.ndarray | float:
    """
    Callable sinc-squared function for fitting.
    
    Y = k*sinc((X-m)/(pi*w))^2 + d.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    k : float
        Scaling coefficient.
    m : float
        Central peak location on X.
    w : float
        Width scaling of central peak.
    d : float
        Constant baseline addition.

    Returns
    -------
    np.ndarray | float
        Y = k*sinc((X-m)/(pi*w))^2 + d.
        
    """
    z = (X - m)/(pi*w)
    return k*(np.sinc(z)**2) + d


def emg_f(X: np.ndarray | float, k: float, m: float, sigma: float, tau: float, d: float) -> np.ndarray | float:
    """
    Callable parametrized exponentially modified Gaussian distribution (EMG) PDF.

    Parameters
    ----------
    X : np.ndarray | float
        Function value(-s).
    k : float
        Scaling parameter.
    m : float
        Center of symmetry.
    sigma : float
        Gaussian width scaling.
    tau : float
        Exponentially decay.
    d : float
        Constant baseline addition.

    Returns
    -------
    np.ndarray | float
        Y = k*scipy.stats.exponnorm(X, K=tau/sigma, loc=m, scale=sigma).
    """
    K = tau / sigma  # as used by SciPy
    return k*exponnorm.pdf(X, K=K, loc=m, scale=sigma) + d


# %% Define peak type and value
def get_peak(f: Callable, fitted_params: tuple[float, ...]) -> tuple[bool, bool, float, float]:
    """
    Get information of a peak (max) or minimum value for the provided function.

    Parameters
    ----------
    f : Callable
        Callable function.
    fitted_params : tuple[float, ...]
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
            a, b, c = fitted_params; is_max = a > 0.0; x0 = b
            if x_min < x0 < x_max and abs(a) >= tol:
                y0 = gaussian_f(x0, a, b, c)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "gaussian_leveled_f":
            a, b, c, d = fitted_params; is_max = a > 0.0; x0 = b
            if x_min < x0 < x_max and abs(a) >= tol:
                y0 = gaussian_leveled_f(x0, a, b, c, d)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "lorentzian_f":
            a, b, k, d = fitted_params; is_max = a*k > 0.0; x0 = b
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = lorentzian_f(x0, a, b, k, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "line_f":
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
        elif f.__name__ == "witch_agnesi_f":
            a, m, k, d = fitted_params; is_max = k > 0.0; x0 = m
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = witch_agnesi_f(x0, a, m, k, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "logistic_derivative_f":
            k, a, b, d = fitted_params; is_max = k > 0.0; x0 = b
            if x_min < x0 < x_max and abs(k) >= tol:
                y0 = logistic_derivative_f(x0, k, a, b, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "cosine_f":
            k, a, b = fitted_params
            # Note - boundaries now should guarantee the presence of only single extremum inside them
            n = round((a*0.5 - b) / pi)  # define an approximation of extreme point definition f'(x) = 0 => a*x - b = pi*n
            x0 = (b + n*pi) / a  # extreme point
            if x_min < x0 < x_max:  # extreme point inside the interval
                is_max = (n % 2 == 0); y0 = k if is_max else -k  # if n - odd, then cos(n*pi) = -1, if even => cos(n*pi) = 1
            else:
                is_definable = False
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
                if is_max:
                    result = minimize_scalar(lambda x: -emg_f(x, *fitted_params), bounds=(0.0, 1.0), method="bounded")
                else:
                    result = minimize_scalar(lambda x: emg_f(x, *fitted_params), bounds=(0.0, 1.0), method="bounded")
                if x_min < result.x < x_max:
                    x0 = result.x; y0 = emg_f(x0, k, m, sigma, tau, d)
                else:
                    is_definable = False  # found extreme point outside or exactly on the X range bounds
            else:
                is_definable = False  # scaling coefficient is close to 0.0
        elif f.__name__ == "cubic_polynomial":
            a, b, c, d = fitted_params; discriminant_dx = b**2 - 3*a*c  # f'(x) = 0 for extreme, f'(x) = 3ax^2 + 2b*x + c
            if discriminant_dx > 0 and abs(a) >= tol:  # two roots - one max, one min
                x01 = (-b + np.sqrt(discriminant_dx))/(3.0*a); x02 = (-b - np.sqrt(discriminant_dx))/(3.0*a)
                x01_in_range = 0.0 < x01 < x_max; x02_in_range = 0.0 < x02 < x_max
                ya = cubic_polynomial(0.0, a, b, c, d); yb = cubic_polynomial(x_max, a, b, c, d)
                # define which peak / valley is global or only local and test for both cases
                is_max_01 = 6.0*a*x01 + 2.0*b < 0.0; y01 = cubic_polynomial(x01, a, b, c, d)
                is_max_02 = 6.0*a*x02 + 2.0*b < 0.0; y02 = cubic_polynomial(x02, a, b, c, d)
                is_global_x01 = x01_in_range and ((is_max_01 and y01 > ya and y01 > yb) or (not is_max_01 and y01 < ya and y01 < yb))
                is_global_x02 = x02_in_range and ((is_max_02 and y02 > ya and y02 > yb) or (not is_max_02 and y02 < ya and y02 < yb))
                # based on defined x1, x2 location provide an estimation of only a peak
                if is_global_x01 and is_global_x02:
                    is_definable = False  # ambiguous for extraction of a single peak / valley
                elif is_global_x01 and not is_global_x02:
                    is_max = is_max_01; x0 = x01; y0 = y01
                elif not is_global_x01 and is_global_x02:
                    is_max = is_max_02; x0 = x02; y0 = y02
                else:
                    is_definable = False
            else:
                is_definable = False  # either there is no max / min, or it's stationary inflection point (f(x) = x^3 it is x = 0)
        elif f.__name__ == "quartic_polynomial":
            a, b, c, d, e = fitted_params; n_digits = 9
            root_tol = 10.0**(-n_digits+3)  # looser tolerance accounting for numerical root-solving uncertainty
            roots = np.roots([4*a, 3*b, 2*c, d])  # for f'(x) = 4*a*x^3 + 3*b*x^2 + 2*c*x + d
            # below - keep only roots with small imaginary part, the returned roots are complex, and real part withing selected x range
            real_roots = sorted([r.real for r in roots if abs(r.imag) < root_tol and x_min < r.real < x_max])
            # keep only unique, distinct roots
            unique_roots = []  # empty container for collecting
            for x_r in real_roots:
                if not unique_roots or abs(x_r - unique_roots[-1]) > root_tol:  # add 1st element or compare with the previous one (max)
                    unique_roots.append(x_r)
            if len(unique_roots) > 0:  # 1, 2 or 3 real, distinguishable roots
                extreme_points = []  # define peak / valley candidates, ignore stationary inflection and flat max / min solutions
                for i, x_r in enumerate(unique_roots):
                    f2 = 12.0*a*x_r**2 + 6.0*b*x_r + 2.0*c  # f''(x_r)
                    if f2 < -tol:  # f''(x_r) < generic tol effectively
                        extreme_points.append({i: "peak"})
                    elif f2 > tol:
                        extreme_points.append({i: "valley"})
                # below - sort out the case of not defined extreme points or 'M' and 'W' like curves as not suitable for peaks retrieval
                if len(extreme_points) == 0 or len(extreme_points) == 3:
                    is_definable = False
                elif len(extreme_points) == 1:
                    y_a = round(quartic_polynomial(x_min, a, b, c, d, e), n_digits)
                    y_b = round(quartic_polynomial(x_max, a, b, c, d, e), n_digits)
                    i_xr = next(iter(extreme_points[0]))  # recorded index of found extreme point
                    xr = unique_roots[i_xr]; y_xr = round(quartic_polynomial(xr, a, b, c, d, e), n_digits)
                    if extreme_points[0][i_xr] == "peak" and y_xr > y_a and y_xr > y_b:
                        is_max = True; x0 = xr; y0 = y_xr
                    elif extreme_points[0][i_xr] == "valley" and y_xr < y_a and y_xr < y_b:
                        is_max = False; x0 = xr; y0 = y_xr
                    else:
                        is_definable = False  # local peak / valley only
                elif len(extreme_points) == 2:  # peak and valley candidates, one of them is only local
                    y_a = round(quartic_polynomial(x_min, a, b, c, d, e), n_digits)
                    y_b = round(quartic_polynomial(x_max, a, b, c, d, e), n_digits)
                    i_xr1 = next(iter(extreme_points[0])); i_xr2 = next(iter(extreme_points[1]))
                    xr1 = unique_roots[i_xr1]; y_xr1 = round(quartic_polynomial(xr1, a, b, c, d, e), n_digits)
                    xr2 = unique_roots[i_xr2]; y_xr2 = round(quartic_polynomial(xr2, a, b, c, d, e), n_digits)
                    is_global_xr1 = ((extreme_points[0][i_xr1] == "peak" and y_xr1 > y_a and y_xr1 > y_b)
                                    or (extreme_points[0][i_xr1] == "valley" and y_xr1 < y_a and y_xr1 < y_b))
                    is_global_xr2 = ((extreme_points[1][i_xr2] == "peak" and y_xr2 > y_a and y_xr2 > y_b)
                                    or (extreme_points[1][i_xr2] == "valley" and y_xr2 < y_a and y_xr2 < y_b))
                    if is_global_xr1 and is_global_xr2:
                        is_definable = False
                    elif is_global_xr1 and not is_global_xr2:
                        is_max = extreme_points[0][i_xr1] == "peak"; x0 = xr1; y0 = y_xr1
                    elif not is_global_xr1 and is_global_xr2:
                        is_max = extreme_points[1][i_xr2] == "peak"; x0 = xr2; y0 = y_xr2
                    else:
                        is_definable = False
                else:
                    is_definable = False
            else:
                is_definable = False

    return is_definable, is_max, x0, y0
