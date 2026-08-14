# -*- coding: utf-8 -*-
"""
Run some tests for peakfit library directly in this script without pytest usage.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
import numpy as np

from peakfitpy import PeakFit2D

plot_all_curves_with_defaults = True
test_simple_case = False


# %% Only for development purposes
if __name__ == "__main__":

    pf = PeakFit2D(x=np.asarray([1, 2, 3]), y=np.asarray([0, 1, 0]))

    # Individual plots with default parameters
    # pf.plot_norm(); pf.plot_norm(x_range="-1,1")  # Gaussian plotting
    # pf.plot_norm(f_name=pf.function_names[1]); pf.plot_norm(f_name=pf.function_names[1], x_range="-1,1")  # Parabola
    # pf.plot_norm(f_name=pf.function_names[2]); pf.plot_norm(f_name=pf.function_names[2], x_range="-1,1")  # Leveled Gaussian
    # pf.plot_norm(f_name=pf.function_names[3]); pf.plot_norm(f_name=pf.function_names[3], x_range="-1,1")  # Lorentzian
    # pf.plot_norm(f_name=pf.function_names[4]); pf.plot_norm(f_name=pf.function_names[4], x_range="-1,1")  # Line (average)
    # pf.plot_norm(f_name=pf.function_names[5]); pf.plot_norm(f_name=pf.function_names[5], x_range="-1,1")  # Hyperbolic secant
    # pf.plot_norm(f_name=pf.function_names[6], x_range="-1,1")  # Bump Function only defined in a range [-b, b] or [-1.0, 1.0]
    # pf.plot_norm(f_name=pf.function_names[7]); pf.plot_norm(f_name=pf.function_names[7], x_range="-1,1")  # Witch of Agnesi
    # pf.plot_norm(f_name=pf.function_names[8]); pf.plot_norm(f_name=pf.function_names[8], x_range="-1,1")  # Deriv. Logistic F()
    # pf.plot_norm(f_name=pf.function_names[9]); pf.plot_norm(f_name=pf.function_names[9], x_range="-1,1")  # Cosine
    # pf.plot_norm(f_name=pf.function_names[10]) # Rayleigh PDF
    # pf.plot_norm(f_name=pf.function_names[11]) # Mirrored Rayleigh PDF
    # pf.plot_norm(f_name=pf.function_names[12]); pf.plot_norm(f_name=pf.function_names[11], x_range="-1,1")  # Laplace PDF

    # Composed plot for both ranges
    if plot_all_curves_with_defaults:
        pf.plot_all_defaults()

    # Test fitting on the simple set
    if test_simple_case:
        pf2 =  PeakFit2D(x=np.asarray([10, 20, 30, 40, 50, 60]), y=1E2*np.asarray([1, 1.5, 2, 2.4, 1.7, 1.24]))
        pf2.fit_function(verbose=True, plot_best_fit=True)
        pf2.fit_function(verbose=True, plot_best_fit=True, x_range="-1,1")

    # Generate some complex examples and visualize the fitting
    x = np.asarray([(1.25*i + 2.2) for i in range(20)]); b = x.mean()
    y = np.exp(-(x - b*1.1)**6/13.0)  + 1.0 / x  # some undefined in a list of implemented functions function
    pf = PeakFit2D(x, y); pf.fit_function(verbose=True, plot_best_fit=True)
