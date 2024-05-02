import numpy as np


def _unitsample_1d(f0, f1, u):
    """Convert uniform samples to single 1D lintsamples on unit interval.
    
    `f0` and `f1` are 1D numpy arrays length N, representing densities at x=0
    and x=1 respectively for N cells. `u` is also a 1D numpy array length N, 
    giving uniform samples U(0, 1) to be converted to N samples from 1D linear
    interpolant on interval (0, 1). This works as follows:
    where f0 == f1: samples = u
    elsewhere: samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)

    Parameters
    ----------
    f0 : 1D numpy array, length N (same length as f1 & u)
        Array of densities at x=0.
    f1 : 1D numpy array, length N (same length as f0 & u)
        Array of densities at x=1.
    u : 1D numpy array, length N (same length as f0 & f1)
        Uniform samples ~U(0,1).

    Returns
    -------
    samples : 1D array, length N
        Samples from cells.
    """
    # where f0 == f1: samples = u
    # elsewhere, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    m = f0 == f1
    z = np.copy(u)
    u = u[~m]
    f0 = f0[~m]
    f1 = f1[~m]
    z[~m] = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z
