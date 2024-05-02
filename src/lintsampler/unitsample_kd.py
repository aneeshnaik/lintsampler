import numpy as np
from math import log2
from .utils import _multiply_array_slice
from .unitsample_1d import _unitsample_1d


def _unitsample_kd_single(*f, u):
    """Single sample from k-linear interpolant in k-dimensional unit hypercube.
    
    f is a series of 2^k scalars, representing the (not necessarily normalised)
    density values at the 2^k corners of the k-dimensional unit hypercube.
    A single sample is drawn from the k-linear interpolant between these
    corners. This works by first 1D sampling from p(x0), then 1D sampling
    from p(x1|x0), then p(x2|x0,x1), and so on until all k dimensions are
    sampled.

    Parameters
    ----------
    *f : 2^k scalars
        Densities at corners of k-d unit cube
    u : numpy array, length k
        Single k-dimensional uniform sample.

    Returns
    -------
    samples : 1D array shape (k)
        Sample from k-linear interpolant.

    """            
    # infer dimensionality
    k = int(log2(len(f)))
    
    # stack densities into cube shape (2 x 2 x ... x 2)
    f = np.stack(f).reshape(((2,) * k))

    # set up empty array for sample
    sample = np.zeros(k)

    # loop over dims, starting with first
    # at each dim sample from p(current dim | previous dims)
    for d in range(k):

        # construct 'reduced' hypercube, averaging over higher dims
        freduced = np.average(f, axis=tuple(range(d + 1, k)))
        
        # loop over dims lower than current
        # at each dim multiply slice 0 by (1-xi) and slice 1 by xi
        for i in range(d):
            _multiply_array_slice(freduced, (1 - sample[i]), i, 0)
            _multiply_array_slice(freduced, sample[i], i, 1)
        
        # sum up to current dim and take slices
        f0, f1 = np.sum(freduced, axis=tuple(range(d)))

        # get samples
        sample[d] = _unitsample_1d(f0, f1, u=u[d])

    return sample


def _unitsample_kd(*f, u):
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
    samples : 1D array shape (k) or 2D array, shape (N, k)
        Sample or batch of samples from k-linear interpolant.

    """    
    # if densities scalar, pass to unbatched function
    if not hasattr(f[0], "__len__"):
        return _unitsample_kd_single(*f, u=u[0])

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
