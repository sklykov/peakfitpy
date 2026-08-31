# -*- coding: utf-8 -*-
"""
Store typing annotations (declarations).

@author: Sergei Klykov, @year: 2026, @license: MIT \n

"""
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

RealScalar = int | float | np.floating[Any] | np.integer[Any]
RealSeq = Sequence[RealScalar]  # for providing type hints accepting types like tuple[float], list[int]
nparray = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]
FloatArray = NDArray[np.floating[Any]]
FitParams = Sequence[float] | NDArray[np.floating[Any]]
