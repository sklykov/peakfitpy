# -*- coding: utf-8 -*-
"""
Test the implemented PeakFit2D class initialization.

The pytest library available on: https://docs.pytest.org/en/latest/contents.html
For running collected here tests, it's enough to run the command "pytest" from the repository location in a command line.

@author: Sergei Klykov
@licence: MIT

"""
import numpy as np

# Importing the written in the modules test functions for letting pytest library their automatic exploration
if __name__ != "__main__":
    from ..peakfit_main import PeakFit2D


def test_class_initialization():
    """
    Test right and wrong specification of input parameters for PeakFit2D initialization.

    Returns
    -------
    None.

    """
    x = np.ones(shape=(5, )); y = np.zeros(shape=(5, ))  # normal vectors
    pkf = PeakFit2D(x_data=x, y_data=y)
    x = np.zeros(shape=(5, 1)); y = np.zeros(shape=(5, ))  # x convertible
    pkf = PeakFit2D(x_data=x, y_data=y)
    x = np.zeros(shape=(5, )); y = np.zeros(shape=(5, 1))  # y convertible
    pkf = PeakFit2D(x_data=x, y_data=y)
    x = np.zeros(shape=(5, 1)); y = np.zeros(shape=(5, 1))  # both convertible
    pkf = PeakFit2D(x_data=x, y_data=y)
    # Testing wrongly sized data
    try:
        x = np.zeros(shape=(5, 1)); y = np.zeros(shape=(5, 2))
        pkf = PeakFit2D(x, y)
    except ValueError:
        pass
    try:
        x = np.zeros(shape=(5, 3)); y = np.zeros(shape=(5, ))
        pkf = PeakFit2D(x, y)
    except ValueError:
        pass
