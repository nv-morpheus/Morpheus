import typing

import cupy as cp
import numpy as np
import numpy.typing as npt

ndarray = typing.Union[cp.ndarray, np.ndarray]

# Once we upgrade to cupy 11+, add in cpt.ArrayLike
ArrayLike = npt.ArrayLike
