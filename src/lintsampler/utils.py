
def _check_N_samples(N_samples):
    """Check requested no. samples is None or positive int, else raise error."""
    if (N_samples is not None):
        if not isinstance(N_samples, int):
            raise TypeError(f"Expected int N_samples, got {type(N_samples)}")
        elif N_samples <= 0:
            raise ValueError(f"Expected positive N_samples, got {N_samples}")
    return

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