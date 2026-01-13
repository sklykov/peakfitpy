# -*- coding: utf-8 -*-
"""
The "peakfitpy" package is intended for fitting 2D curves for further defining of a single peak properties.

@author: Sergei Klykov

@licence: MIT, @year: 206

"""

__version__ = "0.0.1"  # Straightforward way of specifying package version and including it to the package attributes

if __name__ == "__main__":
    # use absolute imports for importing as module
    __all__ = ['peakfit_main']  # for specifying from peakfitpy import * if package imported from some script
elif __name__ == "peakfitpy":
    pass

# Automatically bring the main class and some methods to the name space when one of import command is used commands:
# 1) from peakfitpy import PeakFit2D, ... functions; 2) from peakfitpy import *
if __name__ != "__main__" and __name__ != "__mp_main__":
    from .peakfit_main import PeakFit2D  # main class auto export on the import call of the package
    __all__ = ["PeakFit2D"]
