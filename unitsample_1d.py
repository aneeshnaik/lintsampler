import numpy as np


def unitsample_1d_single(f0, f1, N_samples=None, seed=None):
    """Sample from linear interpolant between x=0 and 1.
    
    f0 and f1 are scalars, representing the (not necessarily normalised)
    densities at x=0 and x=1 respectively. This function draws a sample (or N
    samples) from the linear interpolant between the two points. This works as
    follows: first, u ~ Uniform(0, 1), then:
    if f0 == f1: samples = u
    else: samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)

    Parameters
    ----------
    f0 : scalar
        Density at x=0. 
    f1 : scalar
        Density at x=1.
    N_samples : None/int, optional
        How many samples to draw. If None (default), then single sample (scalar)
        is returned. Otherwise, 1D array of samples is returned.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    sample(s) : scalar or 1D array (length N_samples)
        Sample(s) from linear interpolant. Single sample is returned if
        N_samples is None (default), otherwise 1D array of samples.

    """
    # check f0/f1 scalars
    if hasattr(f0, "__len__") or hasattr(f1, "__len__"):
        raise TypeError("Expected scalar f0/f1.")

    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # generate uniform samples
    u = rng.uniform(size=N_samples)
    
    # if f0 == f1: samples = u
    # else, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    if f0 == f1:
        return u
    z = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z


def unitsample_1d(f0, f1, N_samples=None, seed=None):
    """Batched sampling from 1D linear interpolant between x=0 and 1.
    
    f0 and f1 are numpy arrays length N, representing densities at x=0 and x=1
    respectively for N cells. A single sample (or series of samples) is drawn
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
    N_samples : None/int, optional
        How many samples to draw per cell. If None (default), then single sample
        is drawn in each cell. See below for how this affects shape of output.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    samples : scalar or 1D array or 2D array
        Array of samples from interpolant. Shape and length depends on input
        f0/f1/N_samples as follows:
        f0/f1 scalar, N_samples None: output is scalar
        f0/f1 1D length N, N_samples None: output is 1D array length N
        f0/f1 scalar, N_samples int: output is 1D array length N_samples
        f0/f1 1D length N, N_samples int: output is 2D array (N, N_samples)
    """
    if not hasattr(f0, "__len__"):
        return unitsample_1d_single(f0, f1, N_samples, seed)

    # prepare RNG
    rng = np.random.default_rng(seed)
        
    # get shape of uniform sample array
    N_cells = f0.size
    if N_samples == None:
        size = (N_cells,)
    else:
        size = (N_cells, N_samples)

    # generate uniform samples
    u = rng.uniform(size=size)

    # where f0 == f1: samples = u
    # elsewhere, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    m = f0 == f1
    z = np.copy(u)
    u = u[~m]
    f0 = f0[~m]
    f1 = f1[~m]
    if N_samples != None:
        f0 = f0[:, None]
        f1 = f1[:, None]
    z[~m] = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z
