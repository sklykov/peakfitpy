# -*- coding: utf-8 -*-
"""
Test that the function definitions are consistenly imported in PeakFit1D class.

@author: Sergei Klykov
@license: MIT

"""
# %% Global imports
from peakfitpy import PeakFit1D
from peakfitpy.utils.fitting_funcs import constant_f, default_f_params, full_f_names


# %% Test function
def test_funcs_import():
    """
    Run test on the correct functions import and recognition.

    Returns
    -------
    None
    
    """
    assert len(PeakFit1D.functions) == len(PeakFit1D.function_names), "Not all names of functions exported"
    pf = PeakFit1D(x=[i for i in range(7)], y=[2*i - 0.75 for i in range(7)])
    for f in PeakFit1D.functions:
        assert f.__name__ in pf.function_params, "Not all functions have default parameters for fitting defined"
        assert f.__name__ in default_f_params, "Not all functions have default parameters for fitting defined"
        assert f.__name__ in full_f_names, "Not all functions have full name defined"
    
    assert all(f in PeakFit1D.functions for f in PeakFit1D.polynomials), "Not all polynomials are in the list of functions"
    assert constant_f not in PeakFit1D.polynomials, "Constant line is included in polynomials"
