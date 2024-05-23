import numpy as np
from math import log2
from .utils import _multiply_array_slice, _choice


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


def _unitsample_kd(*f, u):
    # TODO: check docstring (not dealing with scalars anymore)
    """Batched sampling from linear interpolant in k-dimensional unit hypercube.
    
    f is either a series of 2^k scalars or 2^k 1D numpy arrays, each length N,
    representing the (not necessarily normalised) density values at the 2^k
    corners of the k-dimensional unit hypercube (or batch of N such hypercubes).
    A single sample is drawn from the k-linear interpolant between these
    corners. This works by first 1D sampling from p(x0), then 1D sampling
    from p(x1|x0), then p(x2|x0,x1), and so on until all k dimensions are
    sampled.

    Parameters
    ----------
    *f : 2^k scalars or 2^k 1D numpy arrays, length N
        Densities at corners of k-d unit cube (or batch of N such cubes).
    u : 1D numpy array, shape (k,) or 2D numpy array, shape (N, k)
        Single (or N) k-dimensional uniform sample(s).

    Returns
    -------
    samples : 2D array, shape (N, k)
        Batch of samples from k-linear interpolant. If input densities were

    """    
    # infer dimensionality and batch size
    k = int(log2(len(f)))
    N = len(f[0])

    # stack densities into cube shape (2 x 2 x ... x 2 x N)
    f = np.stack(f).reshape(((2,) * k) + (N,))
    
    # set up empty array for samples
    samples = np.zeros((N, k))

    # loop over dims, starting with first
    # at each dim sample from p(current dim | previous dims)
    for d in range(k):

        # construct 'reduced' hypercube, averaging over higher dims
        freduced = np.average(f, axis=tuple(range(d + 1, k)))

        # loop over dims lower than current
        # at each dim multiply slice 0 by (1-xi) and slice 1 by xi
        for i in range(d):     
            _multiply_array_slice(freduced, (1 - samples[:, i]), i, 0)
            _multiply_array_slice(freduced, samples[:, i], i, 1)

        # sum up to current dim and take slices
        f0, f1 = np.sum(freduced, axis=tuple(range(d)))

        # get samples
        samples[:, d] = _unitsample_1d(f0, f1, u=u[:, d])

    return samples


def _grid_sample(grid, u):
    # TODO: docstring
    # TODO: special case of grid with single cell
    
    # get indices of grid cells: 2D array (N, k)
    cells = grid.choose(u[..., -1])

    # get 2^k-tuple of densities at cell corners
    corners = grid.get_cell_corner_densities(cells)

    # sample on unit hypercube
    z = _unitsample_kd(*corners, u=u[..., :-1])

    # rescale coordinates (loop over dimensions)
    for d in range(grid.dim):
        e = grid.edgearrays[d]
        c = cells[:, d]
        z[:, d] = e[c] + np.diff(e)[c] * z[:, d]
    
    return z
