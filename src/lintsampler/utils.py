

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