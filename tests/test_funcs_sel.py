# -*- coding: utf-8 -*-
"""
Test the selection / deselection of functions for fitting loop.

@author: Sergei Klykov
@license: MIT

"""
# %% Global imports
from peakfitpy import PeakFit1D
from peakfitpy.fit_models import constant_f, emg_f, gaussian_f, gaussian_leveled_f


# %% Test function
def test_funcs_selection():
    """
    Run test on the correct functions selection.

    Returns
    -------
    None

    """
    pf = PeakFit1D(x=[i for i in range(8)], y=[2*i - 0.75 for i in range(8)])
    assert len(pf._get_fitting_funcs()) == len(PeakFit1D.functions), "Wrong fallback for internal '_get_fitting_funcs' method"
    selected_funcs = pf._get_fitting_funcs(exclude_funcs=(gaussian_f, gaussian_leveled_f, ))
    assert len(selected_funcs) == len(PeakFit1D.functions)-2, "Wrong excluding of functions"
    selected_funcs = pf._get_fitting_funcs(include_funcs=(constant_f, emg_f))
    assert len(selected_funcs) == 2, "Wrong including of functions"
    pf._get_fitting_funcs(include_funcs=(constant_f, emg_f))
    assert len(pf.best_fit_criteria) == 3 and "IC" in pf.best_fit_criteria, "IC criteria isn't included"
    pf = PeakFit1D(x=[i for i in range(7)], y=[2*i - 0.75 for i in range(7)]); f_crs = pf.best_fit_criteria
    assert len(f_crs) == 2 and "IC" not in f_crs, "IC criteria is included but check of param-s number should prevent it"
    try:
        pf._get_fitting_funcs(include_funcs=())
        raise AssertionError("\nValueError not thrown after providing empty container in 'include_funcs'")
    except ValueError:
        pass
    try:
        pf._get_fitting_funcs(exclude_funcs=PeakFit1D.functions)
        raise AssertionError("\nValueError not thrown after providing all functions for exclusion in 'exclude_funcs'")
    except ValueError:
        pass
