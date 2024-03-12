import numpy as np


def _unitsample_1d_single(f0, f1, seed=None):
    """Sample from linear interpolant between x=0 and 1.
    
    f0 and f1 are scalars, representing the (not necessarily normalised)
    densities at x=0 and x=1 respectively. This function draws a single sample
    from the linear interpolant between the two points. This works as follows:
    first, u ~ Uniform(0, 1), then:
    if f0 == f1: sample = u
    else: sample = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)

    Parameters
    ----------
    f0 : scalar
        Density at x=0. 
    f1 : scalar
        Density at x=1.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    sample : scalar
        Sample from linear interpolant.

    """
    # check f0/f1 scalars
    if hasattr(f0, "__len__") or hasattr(f1, "__len__"):
        raise TypeError("Expected scalar f0/f1.")

    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # generate uniform sample
    u = rng.uniform()
    
    # if f0 == f1: samples = u
    # else, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    if f0 == f1:
        return u
    z = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z


def _unitsample_1d(f0, f1, seed=None):
    """Batched sampling from 1D linear interpolant between x=0 and 1.
    
    f0 and f1 are numpy arrays length N, representing densities at x=0 and x=1
    respectively for N cells. A single sample is drawn for each of the N cells
    from the linear interpolant between the two points. This works as follows:
    first, u ~ Uniform(0, 1), then:
    where f0 == f1: samples = u
    elsewhere: samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)

    Parameters
    ----------
    f0 : scalar or 1D numpy array, length N (same length as f1)
        Density or array of densities at x=0. If scalar, represents density of
        single cell, otherwise densities of batch of cells.
    f1 : scalar or 1D numpy array, length N (same length as f0)
        Density or array of densities at x=1. If scalar, represents density of
        single cell, otherwise densities of batch of cells.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    samples : scalar or 1D array
        Sample(s) from cell(s). Scalar if f0/f1 scalar (representing single
        cell), otherwise 1D array if f0/f1 1D arrays (representing batch of
        cells).
    """
    # if densities scalar, pass to unbatched function
    if not hasattr(f0, "__len__"):
        return _unitsample_1d_single(f0, f1, seed)

    # prepare RNG
    rng = np.random.default_rng(seed)
        
    # get shape of uniform sample array
    N_cells = f0.size

    # generate uniform samples
    u = rng.uniform(size=N_cells)

    # where f0 == f1: samples = u
    # elsewhere, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    m = f0 == f1
    z = np.copy(u)
    u = u[~m]
    f0 = f0[~m]
    f1 = f1[~m]
    z[~m] = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z
