import numpy as np
from scipy.stats.qmc import QMCEngine, Sobol


def _generate_usamples(N, ndim, seed, qmc, qmc_engine):
    """Generate uniform samples, either with RNG or QMC engine."""
    # prepare RNG
    if qmc:
        qmc_engine = _prepare_qmc_engine(qmc_engine, ndim, seed=seed)
        u = qmc_engine.random(N)
    else:
        rng = np.random.default_rng(seed)
        u = rng.random((N, ndim))
    return u


def _choice(p, u):
    """Given probability array and set of uniform samples, return indices."""
    cdf = p.cumsum()
    cdf /= cdf[-1]
    idx = cdf.searchsorted(u, side='right')
    return idx


def _check_N_samples(N_samples):
    """Check requested no. samples is None or positive int, else raise error."""
    if (N_samples is not None):
        if not isinstance(N_samples, int):
            raise TypeError(f"Expected int N_samples, got {type(N_samples)}")
        elif N_samples <= 0:
            raise ValueError(f"Expected positive N_samples, got {N_samples}")
    return


def _prepare_qmc_engine(qmc_engine, k, seed):
    """Parse QMC engine. Scrambled Sobol if None, error if wrong."""
    if qmc_engine is None:
        qmc_engine = Sobol(d=k, scramble=True, bits=30, seed=seed)
    elif isinstance(qmc_engine, QMCEngine):
        if qmc_engine.d != k:
            raise ValueError("Inconsistent engine dimension")
    else:
        raise TypeError("qmc_engine must be QMCEngine instance or None")
    return qmc_engine


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