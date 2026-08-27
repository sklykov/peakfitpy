# -*- coding: utf-8 -*-
"""
Wrap fitting results.

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
# %% Imports
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


# %% Fitting results wrapping class
# Meaning of flags below: make the class attributes not re-assignable, class restricted to own only these param-s, '==' not auto-implemented
@dataclass(frozen=True, slots=True, eq=False)
class Fit1DResult():
    """Dataclass for storing fitting results in its named variables."""
    
    function: Callable
    params: NDArray[np.floating[Any]]  # fitted parameters
    pcov: NDArray[np.floating[Any]] | None  # store report of curve_fit method
    perr: NDArray[np.floating[Any]] | None  # for storing np.sqrt(np.diag(pcov)), all found parameters ~ +- perr
    rmse: float
    mae: float 
    aicc: float | None


# %% Peak defining results wrapping class
@dataclass(frozen=True, slots=True)
class PeakResult():
    """Dataclass for storing fitted peak properties in its named variables."""
    
    is_defined: bool
    is_peak: bool | None
    x: float | None  # for normalized range
    y: float | None  # for normalized range
    fwhm: float | None  # for normalized range
    x_orig : float | None  # for originally scaled range
    y_orig : float | None  # for originally scaled range
    fwhm_orig: float | None  # for originally scaled range
