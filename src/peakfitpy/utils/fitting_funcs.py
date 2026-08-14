# -*- coding: utf-8 -*-
"""
Symbolic definitions of functions for fitting.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
from collections.abc import Callable
from math import e, pi

import numpy as np

default_f_params = {"gaussian_f": {"0,1": [1.0, 0.5, 0.2], "-1,1": [1.0, 0.0, 0.4]},
                    "parabola_f": {"0,1": [-4.0, 4.0, 0.0], "-1,1": [-1.0, 0.0, 1.0]},
                    "gaussian_leveled_f": {"0,1": [1.0, 0.5, 0.15, 0.0], "-1,1": [1.0, 0.0, 0.3, 0.0]},
                    "lorentzian_f": {"0,1": [0.1, 0.5, pi*0.1, 0.0], "-1,1": [0.25, 0.0, pi*0.25, 0.0]},
                    "line_f": {"0,1": [0.0, 0.5], "-1,1": [0.0, 0.5]},
                    "sech_f": {"0,1": [2.0, 8.0, 0.5, 0.0], "-1,1": [2.0, 4.0, 0.0, 0.0]},
                    "bump_f": {"-1,1": [1.001, e, 0.0, 0.0], "0,1": [0.5, e, 0.5, 0.0]},
                    "witch_agnesi_f": {"0,1": [0.1, 0.5, 2.5, 0.0], "-1,1": [0.25, 0.0, 1.0, 0.0]},
                    "logistic_derivative_f": {"0,1": [4.0, 10.0, 0.5, 0.0], "-1,1": [4.0, 5.0, 0.0, 0.0]},
                    "cosine_f": {"0,1": [1.0, pi-1e-4, 0.5*pi], "-1,1": [1.0, 0.5*pi-1e-4, 0.0]},
                    "rayleigh_pdf_f": {"0,1": [0.25, 0.41, 0.0, 0.0], "-1,1": [0.61, 1.0, -1.0, 0.0]},
                    "laplace_pdf_f": {"0,1": [0.5, 0.125, 1.0, 0.0], "-1,1": [0.0, 0.25, 1.0, 0.0]},
                    "rayleigh_pdf_mirrored_f": {"0,1": [0.25, 0.41, 1.0, 0.0], "-1,1": [0.5, 0.824, 1.0, 0.0]},
                    "cubic_polynomial": {"0,1": [-3.472, 1.389, 2.083, 0.0], "-1,1":  [-0.2041, -0.99, 0.2041, 0.99]},
                    "quartic_polynomial":  {"0,1":  [-5.0, 12.472, -13.828, 6.356, 0.0],
                                            "-1,1": [-0.7, 0.0641, -0.2967, -0.0641, 0.9967]},
                    }

full_f_names = {"gaussian_f": "Gaussian", "parabola_f": "Parabola", "gaussian_leveled_f": "Gaussian + Const",
                "lorentzian_f": "Lorentzian", "line_f": "Line", "sech_f": "Hyperbolic Secant", "bump_f": "Bump Function",
                "witch_agnesi_f": "Witch of Agnesi", "logistic_derivative_f": "Derivative of Logistic",
                "cosine_f": "Cosine", "rayleigh_pdf_f": "Rayleigh PDF", "rayleigh_pdf_mirrored_f": "Mirrored Rayleigh PDF",
                "laplace_pdf_f": "Laplace PDF", "cubic_polynomial": "Cubic Polynomial", "quartic_polynomial" : "Quartic Polynomial",}

# Symmetric around the max / min functions
symmetric_f_names = ("gaussian_f", "parabola_f", "gaussian_leveled_f", "lorentzian_f", "sech_f", "bump_f", "witch_agnesi_f",
                     "logistic_derivative_f", "cosine_f", "laplace_pdf_f")

# Generic / assymetric functions
generic_f_names = ("rayleigh_pdf_f", "rayleigh_pdf_mirrored_f", "cubic_polynomial", "quartic_polynomial")

tol = 1e-6  # ultimately is zero for the X and Y ranges laying within [0.0, 1.0] or [-1.0, 1.0]

# Restrictions on fitting parameters for curve_fit method, e.g. for Gaussian: a - not restricted, b - to the padded X range, sigma > tol
# rules: width of function commonly in [tol, +inf), k - unrestricted, b or m (central value) - in padded X range [-1.0, 2.0] or [-2.0, 2.0],
# d - function baseline in padded Y range [-1.0, 2.0]
d_min = -1.0; d_max = 2.0; x01_min = -1.0; x_max = 2.0; xm11_min = -2.0
params_boundaries = {"gaussian_f": {"0,1": ([-np.inf, x01_min, tol], [np.inf, x_max, np.inf]),
                                    "-1,1": ([-np.inf, xm11_min, tol], [np.inf, x_max, np.inf])},
                     "gaussian_leveled_f" : {"0,1": ([-np.inf, x01_min, tol, d_min], [np.inf, x_max, np.inf, d_max]),
                                             "-1,1": ([-np.inf, xm11_min, tol, d_min], [np.inf, x_max, np.inf, d_max])},
                     "lorentzian_f": {"0,1": ([tol, x01_min, -np.inf, d_min], [np.inf, x_max, np.inf, d_max]),
                                      "-1,1": ([tol, xm11_min, -np.inf, d_min], [np.inf, x_max, np.inf, d_max])},
                     "sech_f": {"0,1": ([-np.inf, tol, x01_min, d_min], [np.inf, np.inf, x_max, d_max]),
                                "-1,1": ([-np.inf, tol, xm11_min, d_min], [np.inf, np.inf, x_max, d_max])},
                     "bump_f": {"0,1": ([tol, -np.inf, x01_min, d_min], [np.inf, np.inf, x_max, d_max]),
                                "-1,1": ([tol, -np.inf, xm11_min, d_min], [np.inf, np.inf, x_max, d_max])},
                     "witch_agnesi_f": {"0,1": ([tol, x01_min, -np.inf, d_min], [np.inf, x_max, np.inf, d_max]),
                                        "-1,1": ([tol, xm11_min, -np.inf, d_min], [np.inf, x_max, np.inf, d_max])},
                     "logistic_derivative_f": {"0,1": ([-np.inf, tol, x01_min, d_min], [np.inf, np.inf, x_max, d_max]),
                                               "-1,1": ([-np.inf, tol, xm11_min, d_min], [np.inf, np.inf, x_max, d_max])},
                     "cosine_f": {"0,1": ([tol, tol, -pi], [np.inf, pi-tol, pi-tol]),
                                  "-1,1": ([tol, tol, -pi], [np.inf, 0.5*pi-tol, pi-tol])},
                     "rayleigh_pdf_f": {"0,1": ([tol, -np.inf, x01_min, d_min], [np.inf, np.inf, x_max, d_max]),
                                        "-1,1": ([tol, -np.inf, xm11_min, d_min], [np.inf, np.inf, x_max, d_max])},
                     "rayleigh_pdf_mirrored_f": {"0,1": ([tol, -np.inf, x01_min, d_min], [np.inf, np.inf, x_max, d_max]),
                                                 "-1,1": ([tol, -np.inf, xm11_min, d_min], [np.inf, np.inf, x_max, d_max])},
                     "laplace_pdf_f": {"0,1": ([x01_min, tol, -np.inf, d_min], [x_max, np.inf, np.inf, d_max]),
                                        "-1,1": ([xm11_min, tol, -np.inf, d_min], [x_max, np.inf, np.inf, d_max])},
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
    Parametric Gaussian function with fitting of non-zero level (asympotic minimal Y value) for fitting.

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
        See equation. Constant shift (non-zero) value.

    Returns
    -------
    np.ndarray | float
        Y = a*exp((-(X-b)^2)/2*c^2) + d.

    """
    return a*np.exp(-np.power(X-b, 2)/(2.0*(c**2))) + d


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


# %% Define peak type and value
def get_peak(f: Callable, fitted_params: tuple[float, ...], x_range: str = "0,1") -> tuple[bool, bool, float, float]:
    """
    Get information of a peak (max) or minimum value for the provided function.

    Parameters
    ----------
    f : Callable
        Callable function.
    fitted_params : tuple[float, ...]
        Defined best (fitted) parameters of the function.
    x_range : str, optional
        X range used for fitting. The default is "0,1".

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
    if f.__name__ in default_f_params and x_range in ["0,1", "-1,1"]:
        is_definable = True  # by default function supposed to provide a peak value
        if f.__name__ == "parabola_f":
            a, b, c = fitted_params; is_max = a < 0.0  # parabola opens downward
            if abs(a) >= tol:
                x0 = -(0.5*b)/a  # defined from the 1st derivative
                if (x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0):
                    y0 = parabola_f(x0, a, b, c)
                else:
                    is_definable = False  # peak / valley lays out of provided range
            else:
                is_definable = False  # it's not really a parabola, it's just a line
        elif f.__name__ == "gaussian_f":
            a, b, c = fitted_params; is_max = a > 0.0; x0 = b
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(a) >= tol:
                y0 = gaussian_f(x0, a, b, c)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "gaussian_leveled_f":
            a, b, c, d = fitted_params; is_max = a > 0.0; x0 = b
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(a) >= tol:
                y0 = gaussian_leveled_f(x0, a, b, c, d)
            else:
                is_definable = False  # peak outside the range or a parameter wrongly fitted
        elif f.__name__ == "lorentzian_f":
            a, b, k, d = fitted_params; is_max = a*k > 0.0; x0 = b
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = lorentzian_f(x0, a, b, k, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "line_f":
            k, b = fitted_params; is_max = k > 0.0
            is_definable = abs(k) >= tol  # constant line cannot define if there is peak (min / max) presented
            x0 = 1.0; y0 = k + b  # if is_max - False, then it will be minimum
        elif f.__name__ == "sech_f":
            k, a, b, d = fitted_params; is_max = k > 0.0; x0 = b
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = sech_f(x0, k, a, b, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "bump_f":
            b, k, m, d = fitted_params; is_max = k > 0.0; x0 = m
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = bump_f(x0, b, k, m, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "witch_agnesi_f":
            a, m, k, d = fitted_params; is_max = k > 0.0; x0 = m
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = witch_agnesi_f(x0, a, m, k, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "logistic_derivative_f":
            k, a, b, d = fitted_params; is_max = k > 0.0; x0 = b
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = logistic_derivative_f(x0, k, a, b, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "cosine_f":
            k, a, b = fitted_params
            # Note - boundaries now should guarantee the presence of only single extremum inside them
            if x_range == "0,1":
                x_min, x_max = 0.0, 1.0; x_mid = 0.5
            else:
                x_min, x_max = -1.0, 1.0; x_mid = 0.0
            n = round((a*x_mid - b) / pi)  # define an approximation of extreme point definition f'(x) = 0 => a*x - b = pi*n
            x0 = (b + n*pi) / a  # extreme point
            if x_min <= x0 <= x_max:  # extreme point inside the interval
                is_max = (n % 2 == 0); y0 = k if is_max else -k  # if n - odd, then cos(n*pi) = -1, if even => cos(n*pi) = 1
            else:
                is_definable = False
        elif f.__name__ == "rayleigh_pdf_f":
            s, k, b, d = fitted_params; is_max = k > 0.0; x0 = b + s
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = rayleigh_pdf_f(x0, s, k, b, d)
            else:
                is_definable = False  # peak / valley lays out of provided range or k == 0.0
        elif f.__name__ == "rayleigh_pdf_mirrored_f":
            s, k, b, d = fitted_params; is_max = k > 0.0; x0 = b - s
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = rayleigh_pdf_mirrored_f(x0, s, k, b, d)
            else:
                is_definable = False  # peak / valley lays out of provided range or k == 0.0
        elif f.__name__ == "laplace_pdf_f":
            m, b, k, d = fitted_params; is_max = k > 0.0; x0 = m
            if ((x_range == "0,1" and 0.0 <= x0 <= 1.0) or (x_range == "-1,1" and -1.0 <= x0 <= 1.0)) and abs(k) >= tol:
                y0 = laplace_pdf_f(x0, m, b, k, d)
            else:
                is_definable = False  # peak outside the range or k == 0.0
        elif f.__name__ == "cubic_polynomial":
            a, b, c, d = fitted_params; discriminant_dx = b**2 - 3*a*c  # f'(x) = 0 for extreme, f'(x) = 3ax^2 + 2b*x + c
            if discriminant_dx > 0 and abs(a) >= tol:  # two roots - one max, one min
                x01 = (-b + np.sqrt(discriminant_dx))/(3.0*a); x02 = (-b - np.sqrt(discriminant_dx))/(3.0*a)
                if x_range == "0,1":
                    x01_in_range = 0.0 < x01 < 1.0; x02_in_range = 0.0 < x02 < 1.0
                    ya = cubic_polynomial(0.0, a, b, c, d); yb = cubic_polynomial(1.0, a, b, c, d)
                else:
                    x01_in_range = -1.0 < x01 < 1.0; x02_in_range = -1.0 < x02 < 1.0
                    ya = cubic_polynomial(-1.0, a, b, c, d); yb = cubic_polynomial(1.0, a, b, c, d)
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
            if x_range == "0,1":
                x_min = 0.0; x_max = 1.0
            else:
                x_min = -1.0; x_max = 1.0
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
