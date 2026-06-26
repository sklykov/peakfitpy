# -*- coding: utf-8 -*-
"""
The "peakfitpy" package is intended for fitting 2D curves for possible location of a single peak.

@author: Sergei Klykov

@licence: MIT, @year: 2026

"""

__version__ = "0.0.1"  # Straightforward way of specifying package version and including it to the package attributes

# Univesal logic for making all main classes and functions available after calling 'from project import *'
from .peakfit import PeakFit2D

__all__ = ['PeakFit2D']
