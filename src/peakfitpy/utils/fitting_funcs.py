# -*- coding: utf-8 -*-
"""
Symbolic definitions of functions for fitting.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
from math import pi

import numpy as np

default_f_params = {"gaussian_f": {"0,1": [1.0, 0.5, 0.2], "-1,1": [1.0, 0.0, 0.4]}, 
                    "parabola_f": {"0,1": [-4.0, 4.0, 0.0], "-1,1": [-1.0, 0.0, 1.0]},
                    "gaussian_leveled_f": {"0,1": [1.0, 0.5, 0.15, 0.0], "-1,1": [1.0, 0.0, 0.3, 0.0]},
                    "lorentzian_f": {"0,1": [0.1, 0.5, pi*0.1], "-1,1": [0.25, 0.0, pi*0.25]}, 
                    "line_f": {"0,1": [0.0, 0.5], "-1,1": [0.0, 0.5]}, 
                    "sech_f": {"0,1": [2.0, 8.0, 0.5], "-1,1": [2.0, 4.0, 0.0]}, 
                    "bump_f": {"-1,1": [1.001, 2.0]}, 
                    "witch_agnesi_f": {"0,1": [0.1, 0.5], "-1,1": [0.25, 0.0]}, 
                    "logistic_derivative_f": {"0,1": [4.0, 10.0, 0.5], "-1,1": [4.0, 5.0, 0.0]},
                    "cosine_f": {"0,1": [1.0, pi, 0.5*pi], "-1,1": [1.0, 0.5*pi, 0.0]},
                    "rayleigh_pdf_f": {"0,1": [0.25, 0.4, 0.0]},
                    "laplace_pdf_f":{"0,1": [0.5, 0.125, 1.0], "-1,1": [0.0, 0.25, 1.0]}}

full_f_names = {"gaussian_f": "Gaussian", "parabola_f": "Parabola", "gaussian_leveled_f": "Gaussian + Const", 
                "lorentzian_f": "Lorentzian", "line_f": "Line", "sech_f": "Hyperbolic Secant", "bump_f": "Bump Function", 
                "witch_agnesi_f": "Witch of Agnesi Function", "logistic_derivative_f": "Derivative of Logistic Function",
                "cosine_f": "Cosine", "rayleigh_pdf_f": "Rayleigh PDF", "laplace_pdf_f": "Laplace PDF"}


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

    Equation: a*exp(-(X-b)^2/2*c^2) \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

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


def lorentzian_f(X: np.ndarray, a: float, b: float, k: float) -> np.ndarray:
    """
    Parametric Lorentzian (Cauchy PDF) function for fitting.

    Equation: (a*k) / pi*(a^2 + (X-b)^2) \n 
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    a : float
        See equation (gamma value).
    b : float
        See equation. Mean value (x0).
    k : float
        Scaling coefficient (detached from 'a' or gamma value).

    Returns
    -------
    np.ndarray
        Y = (a*k) / pi*(a^2 + (X-b)^2).

    """
    return (a*k)/(pi*(a**2 + np.power(X-b, 2)))


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
        Y =  k*X + b.

    """
    return k*X + b


def sech_f(X: np.ndarray, k: float, a: float, b: float) -> np.ndarray:
    """
    Parametric hyperbolic secant.
    
    Equation: Y = k / (exp(a*(X-b)) + exp(a*(-X+b))) \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    k : float
        Max value (scaling), for normalized values default value = 2.0 (max at X = 0.0).
    a : float
        Scaling parameter for decay (FWHM).
    b : float
        Offset on X from 0.0 of the peak value.

    Returns
    -------
    np.ndarray
        Y = k / (exp(a*(X-b)) + exp(a*(-X+b))).
    """
    return k / (np.exp(a*(X-b)) + np.exp(a*(-X+b)))


def bump_f(X: np.ndarray, b: float, k: float) -> np.ndarray:
    """
    Parametric bump function.
    
    Equation: Y = k*exp(b^2 / (X^2 - b^2)) where abs(X) < b else 0.0 \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    b : float
        Parameter inside an equation.
    k : float
        Scaling coefficient.

    Returns
    -------
    np.ndarray
        Y = k*exp(b^2 / (X^2 - b^2)) where abs(X) < b else 0.0.
    """
    return np.where(np.abs(X) < b, k*np.exp(b**2 / (X**2 - b**2)), 0.0)


def witch_agnesi_f(X: np.ndarray, a: float, m: float) -> np.ndarray:
    """
    Witch Of Agnesi function.
    
    Equation: Y = 8.0*a^3 / (X^2 + 2.0*a^2) \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    a : float
        Equation parameter, for normalized values default is a = 1/4.
    m : float
        Central shift from 0.0 of peak value in X.

    Returns
    -------
    np.ndarray
        Y = 8.0*a^3 / (X^2 + 2.0*a^2).
    """
    return (8.0*a**3)/((X-m)**2 + 2.0*a**2)


def logistic_derivative_f(X: np.ndarray, k: float, a: float, b: float) -> np.ndarray:
    """
    Parametric derivative of logistic function.
    
    Equation: Y = k*exp(a*(X-b)) / (1.0 + exp(a*(X-b))^2 \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    k : float
        Scaling coefficient, for normalized values default is k = 1/2.
    a : float
        Scaling of X values coefficients.
    b : float
        Shift of the peak from X=0.0

    Returns
    -------
    np.ndarray
        Y = k*exp(a*(X-b)) / (1.0 + exp(a*(X-b)))^2.
    """
    expX = np.exp(a*(X-b))
    return k*(expX / np.power((1.0 + expX), 2))


def cosine_f(X: np.ndarray, k: float, a: float, b: float) -> np.ndarray:
    """
    Parametric cosine function.
    
    Equation: Y = k*cos(a*X). Better to use for fitting on [-1.0, 1.0] interval.

    Parameters
    ----------
    X : np.ndarray
        Function values.
    k : float
        Scaling (amplitude) coefficient.
    a : float
        Phase scaling coefficient.
    b : float
        Phase shift for a peak from X = 0.0

    Returns
    -------
    np.ndarray
        Y = k*cos(a*X-b).
    """
    return k*np.cos(a*X-b)


def rayleigh_pdf_f(X: np.ndarray, sigma: float, k: float, b: float) -> np.ndarray:
    """
    Parametric Rayleigh distribution PDF function.
    
    Equation: Y = (k*|X-b| / sigma^2)*exp(-(X-b)^2/2*sigma^2) \n
    Source: https://en.wikipedia.org/wiki/Rayleigh_distribution

    Parameters
    ----------
    X : np.ndarray
        Function values.
    sigma : float
        Scaling parameter.
    k : float
        Amplitude parameter.
    b : float
        Peak-shift parameter (from peak at X = 0.0)

    Returns
    -------
    np.ndarray
        Y = (k*|X-b| / sigma^2)*exp(-(X-b)^2/2*sigma^2).
    """
    sigma2 = sigma**2; X = np.abs(X - b)
    return ((k*X)/sigma2)*np.exp(-(X**2)/(2.0*sigma2))


def laplace_pdf_f(X: np.ndarray, m: float, b: float, k: float) -> np.ndarray:
    """
    Parametric Laplace distribution PDF function.
    
    Equation: Y = k*exp(-|X-m|/b) \n
    Source: https://en.wikipedia.org/wiki/Laplace_distribution

    Parameters
    ----------
    X : np.ndarray
        Function values.
    m : float
        Mean value.
    b : float
        Scaling parameter.
    k : float
        Amplitude parameter.

    Returns
    -------
    np.ndarray
        Y = k*exp(-|X-m|/b).
    """
    return k*np.exp(-np.abs(X - m)/b)
