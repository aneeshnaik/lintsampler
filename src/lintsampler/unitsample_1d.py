import numpy as np


def _unitsample_1d_single(f0, f1, u):
    """Convert single uniform sample to single 1D lintsample on unit interval.
    
    f0 and f1 are scalars, representing the (not necessarily normalised)
    densities at x=0 and x=1 respectively. u is a single uniform sample ~U(0,1).
    This function converts u into a sample from the linear interpolant between
    the two points. This works as follows:
    if f0 == f1: sample = u
    else: sample = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)

    Parameters
    ----------
    f0 : scalar
        Density at x=0. 
    f1 : scalar
        Density at x=1.
    u : scalar
        Uniform sample ~U(0,1).

    Returns
    -------
    sample : scalar
        Sample from linear interpolant.

    """
    # if f0 == f1: samples = u
    # else, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    if f0 == f1:
        return u
    z = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z


def _unitsample_1d(f0, f1, u):
    """Convert uniform samples to single 1D lintsamples on unit interval.
    
    `f0` and `f1` are either 1D numpy arrays length N or scalars, representing
    densities at x=0 and x=1 respectively for N cells (or a single cell if
    scalar). `u` is also either a 1D numpy array length N or a scalar (should
    match `f0` and `f1`), giving uniform samples U(0, 1) to be converted to N
    samples from 1D linear interpolant on interval (0, 1). This works as 
    follows:
    where f0 == f1: samples = u
    elsewhere: samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)

    Parameters
    ----------
    f0 : scalar or 1D numpy array, length N (same length as f1 & u)
        Density or array of densities at x=0. If scalar, represents density of
        single cell, otherwise densities of batch of cells.
    f1 : scalar or 1D numpy array, length N (same length as f0 & u)
        Density or array of densities at x=1. If scalar, represents density of
        single cell, otherwise densities of batch of cells.
    u : scalar or 1D numpy array, length N (same length as f0 & f1)
        Uniform samples ~U(0,1).

    Returns
    -------
    samples : scalar or 1D array
        Sample(s) from cell(s). Scalar if f0/f1/u scalar (representing single
        cell), otherwise 1D array if f0/f1/u 1D arrays (representing batch of
        cells).
    """
    # if densities scalar, pass to unbatched function
    if not hasattr(f0, "__len__"):
        return _unitsample_1d_single(f0, f1, u)

    # where f0 == f1: samples = u
    # elsewhere, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    m = f0 == f1
    z = np.copy(u)
    u = u[~m]
    f0 = f0[~m]
    f1 = f1[~m]
    z[~m] = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z
