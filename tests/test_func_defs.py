# -*- coding: utf-8 -*-
"""
Test that the function definitions are consistently imported in PeakFit1D class.

@author: Sergei Klykov
@license: MIT

"""
# %% Global imports
from peakfitpy import PeakFit1D
from peakfitpy.fit_models import constant_f
from peakfitpy.utils.model_utils import default_f_params, full_f_names, params_boundaries


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
    pk = "peak"; vk = "valley"
    for f in PeakFit1D.functions:
        f_n = f.__name__
        assert f_n in pf.function_params, "Not all functions have default parameters for fitting defined in the instance variable"
        assert f_n in default_f_params, "Not all functions have default parameters for fitting defined in the property dictionary"
        assert f_n in full_f_names, "Not all functions have full name defined"
        if f_n in params_boundaries:
            tp_limits = type(params_boundaries[f_n]); tp_params = type(default_f_params[f_n])
            if tp_params is dict:
                assert tp_limits is tp_params, f"Function '{f_n}' defines different types for param-s {tp_params} and limits {tp_limits}"
                assert set(default_f_params[f_n]) == {pk, vk}, f"Check for '{f_n}' the {default_f_params[f_n]}"
                assert set(params_boundaries[f_n]) == {pk, vk}, f"Check for '{f_n}' the {params_boundaries[f_n]}"
                assert len(default_f_params[f_n][pk]) == len(default_f_params[f_n][vk]), f"Check for '{f_n}' parameters / limits"
                assert len(params_boundaries[f_n][vk][0]) == len(default_f_params[f_n][vk]), f"Check for '{f_n}' parameters / limits"
                assert len(params_boundaries[f_n][vk][1]) == len(default_f_params[f_n][vk]), f"Check for '{f_n}' parameters / limits"
                assert len(params_boundaries[f_n][pk][0]) == len(default_f_params[f_n][pk]), f"Check for '{f_n}' parameters / limits"
                assert len(params_boundaries[f_n][pk][1]) == len(default_f_params[f_n][pk]), f"Check for '{f_n}' parameters / limits"
            elif tp_params is list:
                assert tp_limits is tuple, f"Function '{f_n}' defines types for limits {tp_limits} - instead of 'tuple'"
            else:
                raise AssertionError(f"Function '{f_n}' defines types for param-s {tp_params} - instead of 'list' or 'dict'")
    
    assert all(f in PeakFit1D.functions for f in PeakFit1D.polynomials), "Not all polynomials are in the list of functions"
    assert constant_f not in PeakFit1D.polynomials, "Constant line is included in polynomials"
