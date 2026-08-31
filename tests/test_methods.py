# -*- coding: utf-8 -*-
"""
Test the correctness of some methods implemented in PeakFit1D.

@author: Sergei Klykov
@license: MIT

"""
import numpy as np

from peakfitpy import PeakFit1D


def test_normalization():
    """
    Test normalization methods.

    Returns
    -------
    None
    
    """
    x = [0.1, 0.2, 0.21, 0.22, 0.23, 0.26, 0.3, 0.4, 0.41, 0.72]
    y = [0.02, -0.1, 0.0, 0.04, 0.12, 0.09, 0.03, 0.0, 0.01, -0.3]
    pf = PeakFit1D(x, y)
    x_norm = pf.normalize_x(x); y_norm = pf.normalize_y(y)
    x_back_denorm = pf.denormalize_x(x_norm)
    y_back_denorm = pf.denormalize_y(y_norm)
    assert np.allclose(x, x_back_denorm) and np.allclose(y, y_back_denorm), "De- and Normalization methods not working properly for arrays"
    assert np.allclose(x_norm, pf.x_norm) and np.allclose(y_norm, pf.y_norm), "Implicit and explicit normalization broken"
    assert np.allclose(x, pf.denormalize_x(pf.x_norm)) and np.allclose(y, pf.denormalize_y(pf.y_norm)), "Check denormalization for x- / y_norm"
    x1 = 0.53; y1 = 0.07
    x1_n = pf.normalize_x(x1); y1_n = pf.normalize_y(y1)
    x1_dn = pf.denormalize_x(x1_n); y1_dn = pf.denormalize_y(y1_n)
    assert np.isclose(x1_dn, x1) and np.isclose(y1, y1_dn), "De- and Normalization methods not working properly for float numbers"
