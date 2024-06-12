import numpy as np


def _get_grid_origin_from_cell_idx(idx, level, dim):
    """Compute the origin of a grid cell in grid coordinates.

    Parameters
    ----------
    idx : int
        The index of the cell in the grid. Must be in the range
        [0, 2**(level * dim)).
    level : int
        The level of the grid. Must be a non-negative integer.
    dim : int
        The dimension of the grid. Must be a positive integer.

    Returns
    -------
    origin : numpy.ndarray
        The integer grid coordinates of the origin of the specified cell in the
        grid. The returned array has a shape of (dim,) and dtype of np.int64.
    """    
    # if at root level, give zeros, otherwise go up a level
    if level == 0:
        return np.zeros(dim, dtype=np.int64)
    else:
        # index of parent cell (by integer division)
        pidx = idx // 2**dim
        
        # orthant of this cell within parent
        orthant = np.unravel_index(idx % 2**dim, [2] * dim)
        
        # recurse
        return 2 * _get_grid_origin_from_cell_idx(pidx, level-1, dim) + orthant


def _get_unit_hypercube_corners(ndim):
    """Get corners of N-dimensional unit hypercube.
    
    An N-dimensional unit hypercube has 2**N corners. This function returns
    the coordinates of those corners, starting at the origin then iteratively
    incrementing along each dimension *from last to first*. See example below
    for illustration.

    Parameters
    ----------
    ndim : int
        Number of dimensions of hypercube.

    Returns
    -------
    corners : numpy array, shape (2**ndim, ndim)
        Coordinates of corners.
    
    Example
    -------
    >>> get_unit_hypercube_corners(3)
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 0, 0],
           [1, 0, 1],
           [1, 1, 0],
           [1, 1, 1]])
    """
    # no. corners
    Nc = 2**ndim
    
    # get binary rep of integers from 0 to Nc, split into digits, stack
    c = np.stack(
        [[int(i) for i in np.binary_repr(k, width=ndim)] for k in range(Nc)]
    )
    return c


def _get_hypercube_corners(mins, maxs):
    """Get corners of N-d hypercube with given coordinate minima and maxima.
    
    An N-dimensional unit hypercube has 2**N corners. This function returns
    the coordinates of those corners, starting at the origin then iteratively
    incrementing along each dimension *from last to first*. See example below
    for illustration.

    Parameters
    ----------
    mins : 1D array-like, length N
        Coordinate minima of N-d hypercube.
    maxs : 1D array-like, length N
        Coordinate maxima of N-D hypercube.

    Returns
    -------
    corners : numpy array, shape (2**N, N), dtype same as input mins/maxs
        Coordinates of corners.
    
    Example
    -------
    >>> get_hypercube_corners([10., 20., 35.], [60., 27., 81.])
    array([[10., 20., 35.],
           [10., 20., 81.],
           [10., 27., 35.],
           [10., 27., 81.],
           [60., 20., 35.],
           [60., 20., 81.],
           [60., 27., 35.],
           [60., 27., 81.]])
    """
    # cast to numpy arrays if not already
    mins = np.array(mins)
    maxs = np.array(maxs)
    
    # infer dimensionality
    ndim = len(mins)
    
    # get unit cube corners and scale
    corners = mins + (maxs-mins) * _get_unit_hypercube_corners(ndim)
    return corners


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