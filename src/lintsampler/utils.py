import numpy as np
from scipy.stats.qmc import QMCEngine, Sobol


def _is_1D_iterable(arr):
    # TODO: docstring
    return hasattr(arr, "__len__") and not hasattr(arr[0], "__len__")


def _all_are_instances(iterable, type):
    """Check if all elements in an iterable are instances of specified type.

    Parameters
    ----------
    iterable : iterable
        The iterable whose elements are to be checked.
    type : type
        The type against which each element in the iterable is checked.

    Returns
    -------
    bool
        True if all elements in the iterable are instances of specified type,
        False otherwise.
    """
    return all([isinstance(item, type) for item in iterable])


def _check_hyperbox_overlap(A_mins, A_maxs, B_mins, B_maxs):
    """
    Test whether two hyperboxes A and B overlap.

    Parameters
    ----------
    A_mins : array-like, 1D
        Minimum coordinates of hyperbox A.
    A_maxs : array-like
        Maximum coordinates of hyperbox A.
    B_mins : array-like
        Minimum coordinates of hyperbox B.
    B_maxs : array-like
        Maximum coordinates of hyperbox B.

    Returns
    -------
    bool
        True if hyperboxes A and B overlap, False otherwise.
    """
    return not np.any((A_maxs <= B_mins) | (B_maxs <= A_mins))

def _choice(p, u, return_cdf=False):
    """Given probability array and set of uniform samples, return indices."""
    cdf = p.cumsum()
    cdf /= cdf[-1]
    idx = cdf.searchsorted(u, side='right')
    if return_cdf:
        return idx, cdf
    return idx


def _multiply_array_slice(arr, factor, axis, idx):
    """Multiply (inplace) subslice of array by given factor along given slice.
    
    Parameters
    ----------
    arr : n-dimensional numpy array
        Array to be multiplied.
    factor : float or array broadcastable with given slice
        Factor to multiply array slice by.
    axis : int
        Dimension along which to take slice
    idx : int
        Index at which to take slice along specified axis.
    
    Returns
    -------
    None

    """
    I = [slice(None)] * arr.ndim
    I[axis] = idx
    arr[tuple(I)] *= factor
    return