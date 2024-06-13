import numpy as np
from math import log2
from .utils import _multiply_array_slice, _choice


def _unitsample_1d(f0, f1, u):
    """Convert uniform samples to 1D lintsamples on unit interval.
    
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
    """Convert uniform samples to kD lintsamples in unit hypercube.
    
    f is either a series 2^k 1D numpy arrays, each length N, representing the
    (not necessarily normalised) density values at the 2^k corners of a batch of
    N k-dimensional unit hypercubes. A single sample is drawn from the k-linear
    interpolant between these corners. This works by first 1D sampling from
    p(x0), then 1D sampling from p(x1|x0), then p(x2|x0,x1), and so on until all
    k dimensions are sampled.

    Parameters
    ----------
    *f : 2^k 1D numpy arrays, length N
        Densities at the 2^k corners of N k-dimensional unit cubes. Corners
        should be ordered e.g., in 3D, (0, 0, 0), (0, 0, 1), (0, 1, 0), ...
        (1, 1, 0), (1, 1, 1).
    u : 2D numpy array, shape (N, k)
        Batch of N k-dimensional uniform samples.

    Returns
    -------
    samples : 2D array, shape (N, k)
        Batch of samples from k-linear interpolant.
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
    """
    Sample points from a DensityGrid instance using batch of uniform samples.

    Parameters
    ----------
    grid : DensityGrid
        An instance of the DensityGrid class, which defines the density
        distribution and grid structure. The grid should already have densities
        evaluated, i.e. grid.densities_evaluated should be True.
    u : numpy array
        A 2D array of shape (N, k+1) containing uniform samples. Each row
        represents  a sample in the (k+1)-dimensional uniform sample; the
        last dimension is used to choose a grid cell and the first k dimensions
        are used for sampling within the chosen cell.

    Returns
    -------
    z : ndarray
        A 2D array of shape (N, k) containing the sampled points. Each row
        represents a sampled point in k-dimensional space.
    """
    # choose cells, get mins/maxs/corner densities
    mins, maxs, corners = grid.choose_cells(u[..., -1])

    # sample on unit hypercube
    z = _unitsample_kd(*corners, u=u[..., :-1])

    # rescale coordinates
    z = mins + (maxs - mins) * z
    return z
