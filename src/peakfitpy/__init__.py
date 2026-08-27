# -*- coding: utf-8 -*-
"""
The "peakfitpy" package provides automatic model fitting and single peak/valley characterization for 1D sampled data.

@author: Sergei Klykov

@license: MIT, @year: 2026

"""

__version__ = "0.0.1"  # Straightforward way of specifying package version and including it to the package attributes

# Universal logic for making all main classes and functions available after calling 'from peakfitpy import *'
from .peakfit import PeakFit1D
from .utils.fitting_result import Fit1DResult, PeakResult

__all__ = ['PeakFit1D', 'Fit1DResult', 'PeakResult']
