# -*- coding: utf-8 -*-
"""
Symbolic definitions of functions for fitting.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
import numpy as np


def parabola_f(X: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Callable parabola function for fitting.

    Parameters
    ----------
    X : np.ndarray
        Function values.
    a : float
        Coefficient on x^2.
    b : float
        Coefficient on x.
    c : float
        Coefficient on x^0.

    Returns
    -------
    np.ndarray
        Y = a*X^2 + b*X + c.

    """
    return a*X*X + b*X + c


def gaussian_f(X: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Parametric Gaussian function for fitting with zero asymptotic minimal Y value.

    Equation: a*exp(-(X-b)^2/2*c^2).

    Parameters
    ----------
    X : np.ndarray
        Function values.
    a : float
        See equation.
    b : float
        See equation. Mean value.
    c : float
        See equation. Sigma value.

    Returns
    -------
    np.ndarray
        Y = a*exp(-(X-b)^2/2*c^2).

    """
    return a*np.exp(-(np.power(X-b, 2))/(2.0*(c**2)))


def gaussian_leveled_f(X: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Parametric Gaussian function with fitting of non-zero level (asympotic minimal Y value) for fitting.

    Equation: a*exp(-(x-b)^2/2*c^2) + d.

    Parameters
    ----------
    X : np.ndarray
        Function values.
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
    np.ndarray
        Y = a*exp(-(X-b)^2/2*c^2) + d.

    """
    return a*np.exp(-np.power(X-b, 2)/(2.0*(c**2))) + d


def lorentzian_f(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Parametric Lorentzian (Cauchy PDF) function for fitting.

    Equation: a/(a^2 + (X-b)^2).

    Parameters
    ----------
    X : np.ndarray
        Function values.
    a : float
        See equation (gamma value).
    b : float
        See equation. Mean value (x0).

    Returns
    -------
    np.ndarray
        Y = a/(a^2 + (X-b)^2).

    """
    return a/(a**2 + np.power(X-b, 2))


def line_f(X: np.ndarray, k: float, b: float)-> np.ndarray:
    """
    Parametric Line function for fallback fitting.

    Equation: k*X + b.

    Parameters
    ----------
    X : np.ndarray
        Function values.
    k : float
        Linear coefficient.
    b : float
        y = f(0) value.

    Returns
    -------
    np.ndarray
        Y = a/(a^2 + (X-b)^2).

    """
    return k*X + b
