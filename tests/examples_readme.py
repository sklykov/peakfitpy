# -*- coding: utf-8 -*-
"""
Prepare examples and plots for README.

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
import matplotlib.pyplot as plt
import numpy as np

from peakfitpy import PeakFit1D
from peakfitpy.fit_models import emg_f

x = np.linspace(-5.0, 5.0, 101)
y = 2.0 + 4.0*np.exp(-((x - 1.4)**2)/(2.0*0.8**2))
y = PeakFit1D.add_awgn(y, noise_fraction=0.028, seed=25)  # Repeatable Gaussian noise

fitter = PeakFit1D(x, y)
fit, peak = fitter.find_best_fit(selection_criterion="IC", exclude_funcs=(emg_f,))  # Compare fits using AICc

if fit is not None:
    print("Selected function:", fit.function.__name__)
    print("Normalized RMSE:", fit.rmse)
    y_fitted = fitter.interpolate_y(x)  # Evaluate in the original X and Y units
    fitter.plot_best_curve()
    plt.show()

if peak is not None:
    print("Peak" if peak.is_peak else "Valley")
    print("Coordinates:", peak.x_orig, peak.y_orig)
    print("FWHM:", peak.fwhm_orig)  # None when this model has no implemented FWHM
