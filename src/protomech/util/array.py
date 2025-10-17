"""Array utility functions."""

import numpy as np
from numpy.typing import ArrayLike


def float_equal(arr1: ArrayLike, arr2: ArrayLike) -> bool:
    """Check if two arrays are equal within floating point precision.

    :param arr1: First array
    :param arr2: Second array
    :return: Boolean
    """
    arr1_ = np.asarray(arr1)
    arr2_ = np.asarray(arr2)
    return arr1_.shape == arr2_.shape and np.allclose(arr1_, arr2_)
