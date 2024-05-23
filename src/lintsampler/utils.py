import numpy as np


def _is_1D_iterable(obj):
    """Check if object is 1D iterable (has length but first element doesn't).
    
    Parameters
    ----------
    obj : any
        Object to check
    
    Returns
    is_1D_iterable : bool
        True if object is 1D iterable (is iterable but first element isn't),
        False otherwise.
    """
    return hasattr(obj, "__len__") and not hasattr(obj[0], "__len__")


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


def _choice(p, u):
    """Given probability array and set of uniform samples, return indices.
    
    Parameters
    ----------
    p : array
        1D array of probabilities, shape (N_idx,)
    u : array
        1D array of uniform samples, ~U(0,1), shape (N_u,)
    
    Returns
    -------
    idx : array
        1D array of integers These correspond to indices from the probability
        array `p`. Shape (N_u,)
    """
    cdf = p.cumsum()
    cdf /= cdf[-1]
    idx = cdf.searchsorted(u, side='right')
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