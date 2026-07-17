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


def lorentzian_f(X: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Parametric Lorentzian (Cauchy PDF) function for fitting.

    Equation: a/(a^2 + (X-b)^2) \n 
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

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
        Y =  k*X + b.

    """
    return k*X + b


def sech_f(X: np.ndarray, m: float) -> np.ndarray:
    """
    Parametric hyperbolic secant.
    
    Equation: Y = m / (exp(X) + exp(-X)) \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    m : float
        Max value (scaling), for normalized values default value = 2.0 (max at X = 0.0).

    Returns
    -------
    np.ndarray
        Y = m / (exp(X) + exp(-X)).
    """
    return m / (np.exp(X) + np.exp(-X))


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


def witch_agnesi_f(X: np.ndarray, a: float) -> np.ndarray:
    """
    Witch Of Agnesi function.
    
    Equation: Y = 8.0*a^3 / (X^2 + 2.0*a^2) \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    a : float
        Equation parameter, for normalized values default is a = 1/2.

    Returns
    -------
    np.ndarray
        Y = 8.0*a^3 / (X^2 + 2.0*a^2).
    """
    return (8.0*a**3)/(X**2 + 2.0*a**2)


def logistic_derivative_f(X: np.ndarray, k: float) -> np.ndarray:
    """
    Parametric derivative of logistic function.
    
    Equation: Y = k*exp(X) / (1.0 + exp(X))^2 \n
    Source: https://en.wikipedia.org/wiki/Bell-shaped_function

    Parameters
    ----------
    X : np.ndarray
        Function values.
    k : float
        Scaling coefficient, for normalized values default is k = 1/2.

    Returns
    -------
    np.ndarray
        Y = k*exp(X) / (1.0 + exp(X))^2.
    """
    expX = np.exp(X)
    return k*(expX / np.power((1.0 + expX), 2))


def cosine_f(X: np.ndarray, k: float, a:float) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Y = k*cos(a*X).
    """
    return k*np.cos(a*X)


def rayleigh_pdf_f(X: np.ndarray, sigma: float, k: float) -> np.ndarray:
    """
    Parametric Rayleigh distribution PDF function.
    
    Equation: Y = (k*X / sigma^2)*exp(-X^2/2*sigma^2) \n
    Source: https://en.wikipedia.org/wiki/Rayleigh_distribution

    Parameters
    ----------
    X : np.ndarray
        Function values.
    sigma : float
        Scaling parameter.
    k : float
        Amplitude parameter.

    Returns
    -------
    np.ndarray
        Y = (k*X / sigma^2)*exp(-X^2/2*sigma^2).
    """
    sigma2 = sigma**2
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
