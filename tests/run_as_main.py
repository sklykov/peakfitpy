# -*- coding: utf-8 -*-
"""
Run some tests for peakfit library directly in this script without pytest usage.

@author: Sergei Klykov, @year: 2026, @licence: MIT \n

"""
import numpy as np

from peakfitpy import PeakFit2D

# %% Only for development purposes
if __name__ == "__main__":
    pf = PeakFit2D(x=np.asarray([1, 2, 3]), y=np.asarray([0, 1, 0]))
    # pf.plot_norm(); pf.plot_norm(x_range="-1,1")  # Gaussian plotting
    # pf.plot_norm(f_name=pf.function_names[1]); pf.plot_norm(f_name=pf.function_names[1], x_range="-1,1")  # Parabola 
    # pf.plot_norm(f_name=pf.function_names[2]); pf.plot_norm(f_name=pf.function_names[2], x_range="-1,1")  # Leveled Gaussian 
    # pf.plot_norm(f_name=pf.function_names[3]); pf.plot_norm(f_name=pf.function_names[3], x_range="-1,1")  # Lorentzian
    # pf.plot_norm(f_name=pf.function_names[4]); pf.plot_norm(f_name=pf.function_names[4], x_range="-1,1")  # Line (average)
    # pf.plot_norm(f_name=pf.function_names[5]); pf.plot_norm(f_name=pf.function_names[5], x_range="-1,1")  # Hyperbolic secant
    # pf.plot_norm(f_name=pf.function_names[6], x_range="-1,1")  # Bump Function only defined in a range [-b, b] or [-1.0, 1.0]
    pf.plot_norm(f_name=pf.function_names[7]); pf.plot_norm(f_name=pf.function_names[7], x_range="-1,1")  # Witch of Agnesi
