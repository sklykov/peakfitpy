# -*- coding: utf-8 -*-
"""
Symbolic definitions of functions for fitting + associated properties calculation.

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
from math import pi

import numpy as np
from scipy.stats import exponnorm

from .utils.typing_utils import FloatArray


# %% Function def-s
def parabola_f(X: FloatArray | float, a: float, b: float, c: float) -> FloatArray | float:
    """
    Callable parabola function for fitting.

    Equation: Y = a*X^2 + b*X + c

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    a : float
        Coefficient on x^2.
    b : float
        Coefficient on x.
    c : float
        Coefficient on x^0.

    Returns
    -------
    FloatArray | float
        Y = a*X^2 + b*X + c.

    """
    return a*X*X + b*X + c


def gaussian_f(X: FloatArray | float, k: float, b: float, c: float) -> FloatArray | float:
    """
    Parametric Gaussian function for fitting with zero asymptotic minimal Y value.

    Equation: k*exp((-(X-b)^2)/(2*c^2)) \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    k : float
        See equation.
    b : float
        See equation. Mean value.
    c : float
        See equation. Sigma value.

    Returns
    -------
    FloatArray
        Y = k*exp((-(X-b)^2)/(2*c^2)).

    """
    return k*np.exp(-(np.power(X-b, 2))/(2.0*(c**2)))


def gaussian_leveled_f(X: FloatArray | float, k: float, b: float, c: float, d: float) -> FloatArray | float:
    """
    Parametric Gaussian function with fitting of non-zero level (asymptotic minimal Y value).

    Equation: k*exp((-(X-b)^2)/(2*c^2)) + d.

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    k : float
        See equation.
    b : float
        See equation. Mean value.
    c : float
        See equation. Sigma value.
    d : float
        Constant level.

    Returns
    -------
    FloatArray | float
        Y = k*exp((-(X-b)^2)/(2*c^2)) + d.

    """
    return k*np.exp(-np.power(X-b, 2)/(2.0*(c**2))) + d


def generalized_gaussian_f(X: FloatArray | float, w: float, st: float, m: float, k: float, d: float) -> FloatArray | float:
    """
    Parametric generalized (with arbitrary within exp) Gaussian function.

    Equation: k*exp(-|(X-m)/w|^st) + d.

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = k*exp(-|(X-m)/w|^st) + d.

    """
    z = np.abs((X-m)/w)
    return k*np.exp(-np.power(z, st)) + d


def lorentzian_f(X: FloatArray | float, a: float, b: float, k: float, d: float) -> FloatArray | float:
    """
    Parametric Lorentzian (Cauchy PDF) function for fitting.

    Equation: ((a*k) / (pi*(a^2 + (X-b)^2))) + d \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    a : float
        See equation (gamma value).
    b : float
        See equation. Symmetry center (x0).
    k : float
        Scaling coefficient (detached from 'a' or gamma value).
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    FloatArray | float
        Y = ((a*k) / (pi*(a^2 + (X-b)^2))) + d .

    """
    return ((a*k)/(pi*(a**2 + np.power(X-b, 2)))) + d


def line_f(X: FloatArray | float, k: float, b: float)-> FloatArray | float:
    """
    Parametric Line function for fallback fitting.

    Equation: k*X + b.

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    k : float
        Linear coefficient.
    b : float
        y = f(0) value.

    Returns
    -------
    FloatArray | float
        Y =  k*X + b.

    """
    return k*X + b


def constant_f(X: FloatArray | float, b: float) ->  FloatArray | float:
    """
    Parametric Constant Line function for fallback fitting of values without clear extreme point.

    Equation: Y = 0.0*X + b.

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s), placeholder.
    b : float
        y = f(0) value.

    Returns
    -------
    FloatArray | float
        Y = 0.0*X + b.

    """
    return line_f(X, 0.0, b)


def sech_f(X: FloatArray | float, k: float, a: float, b: float, d: float) -> FloatArray | float:
    """
    Parametric hyperbolic secant.

    Equation: Y = k*exp(-|(X-b)/a|) / (exp(-2.0*|(X-b)/a|) + 1.0) + d \n

    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    k : float
        Scaling parameter.
    a : float
        Width parameter of distribution.
    b : float
        Offset on X from 0.0 of the peak value.
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    FloatArray | float
        Y = k*exp(-|(X-b)/a|) / (exp(-2.0*|(X-b)/a|) + 1.0) + d.

    """
    z = np.abs((X-b)/a); exp_z = np.exp(-z)  # better for numerical stability, trick is to avoid computation of exp(z), z = huge
    return ((k*exp_z)/(1.0 + exp_z**2)) + d


def bump_f(X: FloatArray | float, b: float, k: float, m: float, d: float) -> FloatArray | float:
    """
    Parametric bump function.

    Equation: Y = k*exp(b^2 / ((X-m)^2 - b^2)) + d where abs(X-m) < b else d \n

    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
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


def logistic_derivative_f(X: FloatArray | float, k: float, a: float, b: float, d: float) -> FloatArray | float:
    """
    Parametric derivative of logistic function.

    Equation: Y = k*(exp(-|(X-b)/a|) / (1.0 + exp(-|(X-b)/a|))^2) + d \n

    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    k : float
        Scaling coefficient, for normalized values default is k = 4.0 for unit height.
    a : float
        Width scaling parameter.
    b : float
        Shift of the peak from X=0.0
    d : float
        Constant level if measurement tails are constant (not decay to zero).

    Returns
    -------
    FloatArray | float
        Y = k*(exp(-|(X-b)/a|) / (1.0 + exp(-|(X-b)/a|))^2) + d.

    """
    z = np.abs((X-b)/a); exp_z = np.exp(-z)
    return k*(exp_z / np.power((1.0 + exp_z), 2)) + d


def rayleigh_pdf_f(X: FloatArray | float, sigma: float, k: float, b: float, d: float) -> FloatArray | float:
    """
    Parametric Rayleigh distribution PDF function.

    Equation: Y = (k*(X-b) / sigma^2)*exp(-(X-b)^2/(2*sigma^2)) + d for x >= b else d  \n
    Source: https://en.wikipedia.org/wiki/Rayleigh_distribution

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = (k*(X-b) / sigma^2)*exp(-(X-b)^2/(2*sigma^2)) + d for x >= b else d.

    """
    sigma2 = sigma**2; X_shifted = X - b
    return np.where(X_shifted >= 0.0, ((k*X_shifted)/sigma2)*np.exp(-(X_shifted**2)/(2.0*sigma2)) + d, d)


def rayleigh_pdf_mirrored_f(X: FloatArray | float, sigma: float, k: float, b: float, d: float) -> FloatArray | float:
    """
    Parametric mirrored Rayleigh distribution PDF function.

    Equation: Y = (k*(b-X) / sigma^2)*exp(-(b-X)^2/(2*sigma^2)) + d for x <= b else d  \n
    Source: https://en.wikipedia.org/wiki/Rayleigh_distribution

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = (k*(b-X) / sigma^2)*exp(-(b-X)^2/(2*sigma^2)) + d for x <= b else d .

    """
    sigma2 = sigma**2; X_shifted = b - X
    return np.where(X_shifted >= 0.0, ((k*X_shifted)/sigma2)*np.exp(-(X_shifted**2)/(2.0*sigma2)) + d, d)


def laplace_pdf_f(X: FloatArray | float, m: float, b: float, k: float, d: float) -> FloatArray | float:
    """
    Parametric Laplace distribution PDF function.

    Equation: Y = k*exp(-|X-m|/b) + d \n
    Source: https://en.wikipedia.org/wiki/Laplace_distribution

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = k*exp(-|X-m|/b) + d.

    """
    return k*np.exp(-np.abs(X - m)/b) + d


def quartic_polynomial(X: FloatArray | float, a: float, b: float, c: float, d: float, e: float) -> FloatArray | float:
    """
    Callable quartic polynomial function for fitting.

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = a*X^4 + b*X^3 + c*X^2 + d*X + e.

    """
    return a*(X**4) + b*(X**3) + c*(X**2) + d*X + e


def cubic_polynomial(X: FloatArray | float, a: float, b: float, c: float, d: float) -> FloatArray | float:
    """
    Callable cubic polynomial function for fitting.

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = a*X^3 + b*X^2 + c*X + d.

    """
    return a*(X**3) + b*(X**2) + c*X + d


def moffat_f(X: FloatArray | float, k: float, m: float, w: float, beta: float, d: float) -> FloatArray | float:
    """
    Callable Moffat function for fitting.

    Function Y = k*((1.0 + ((X-m)/w)^2)^-beta) + d. \n

    Reference: https://en.wikipedia.org/wiki/Moffat_distribution

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = k*((1.0 + ((X-m)/w)^2)^-beta) + d.

    """
    z = (X - m)/w; zb = (1.0 + z**2)**(-beta)
    return k*zb + d


def sinc_sq_f(X: FloatArray | float, k: float, m: float, w: float, d: float) -> FloatArray | float:
    """
    Callable sinc-squared function for fitting.

    Y = k*sinc((X-m)/(pi*w))^2 + d.

    Parameters
    ----------
    X : FloatArray | float
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
    FloatArray | float
        Y = k*sinc((X-m)/(pi*w))^2 + d.

    """
    z = (X - m)/(pi*w)
    return k*(np.sinc(z)**2) + d


def emg_f(X: FloatArray | float, k: float, m: float, sigma: float, tau: float, d: float) -> FloatArray | float:
    """
    Callable parametrized exponentially modified Gaussian distribution (EMG) PDF.

    Y = k*scipy.stats.exponnorm.pdf(X, K=tau/sigma, loc=m, scale=sigma) + d.

    Parameters
    ----------
    X : FloatArray | float
        Function value(-s).
    k : float
        Scaling parameter.
    m : float
        Gaussian-component location parameter in the SciPy parameterization.
    sigma : float
        Gaussian width scaling.
    tau : float
        Exponential decay scale.
    d : float
        Constant baseline addition.

    Returns
    -------
    FloatArray | float
        Y = k*scipy.stats.exponnorm.pdf(X, K=tau/sigma, loc=m, scale=sigma) + d.

    """
    K = tau / sigma  # as used by SciPy
    return k*exponnorm.pdf(X, K=K, loc=m, scale=sigma) + d
