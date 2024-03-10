import numpy as np
from math import log2
from .unitsample_kd import unitsample_kd
from .unitsample_1d import unitsample_1d


def sample(x0, x1, *f, N_samples=None, seed=None):
    """Draw sample(s) from k-D hyperbox(es) with known vertex densities.

    Given a k-dimensional hyperbox (or a set of such boxes) with densities known
    only at the 2^k corners of the box(es), draw a sample (or series of samples)
    via interpolant sampling. If a series of boxes, first estimate mass(es) via 
    trapezoid rule then randomly choose from mass-weighted list.
    
    Parameters
    ----------
    x0 : scalar or 1D numpy array, length k, or 2D numpy array, shape (N, k) 
        Coordinates of 'first' cell corner. Scalar if considering
        single 1D cell. 1D array if considering single k-D cell. 2D
        array if considering series of N k-D cells. Note: if considering
        series of 1D cells, need shape (N, 1), not (N).
    x1 : 1D numpy array, length N, or 2D numpy array, shape (N, k)
        As x0, but coordinates of 'last' cell corner.
    *f : 2^k scalars or 2^k 1D numpy arrays, length N.
        Densities at cell corners. Each f should be scalar is considering single
        cell or 1D array if considering series of cells.
    N_samples : int, optional
        Number of samples to draw. Default is None, in which case a single
        sample is drawn.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    X : scalar, 1D array (length k OR N_samples) or 2D array (N_samples, k)
        Sample(s) from linear interpolant. Scalar if single sample (i.e.,
        N_samples is None) in 1D. 1D array if single sample in k-D OR multiple
        samples in 1D. 2D array if multiple samples in k-D.
    """    
    # if x0/x1 scalar (1D coords) then promote to single-element 1D array
    if not hasattr(x0, "__len__"):
        x0 = np.array([x0])
        x1 = np.array([x1])

    # if x0/x1 1D  (no batching) then add singleton batch dimension
    if x0.ndim == 1:
        x0 = x0[None]
        x1 = x1[None]

    # if densities scalars (no batching), promote to single-element 1D array
    if not hasattr(f[0], "__len__"):
        f = tuple([np.array([fi]) for fi in f])

    # check x0/x1 have same shape
    if x0.shape != x1.shape:
        raise ValueError("x0/x1 have different shapes")

    # infer dimensionality and batch size
    Nb, k = x0.shape
    
    # check appropriate number of densities given
    if len(f) != 2**k:
        raise ValueError("Expected 2^k corner densities.")

    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # choose cell(s) 
    c = _cell_choice(x0, x1, *f, N_cells=N_samples, seed=rng)

    # subset coords and densities of chosen cell(s)
    x0 = x0[c]
    x1 = x1[c]
    f = tuple([fi[c] for fi in f])
    
    # draw samples
    z = x0 + (x1 - x0) * unitsample_kd(*f, seed=rng)

    # squeeze down to scalar / 1D if appropriate
    if not N_samples and (k == 1):
        z = z.item()
    elif (k == 1):
        z = z[:, 0]

    return z
    

def _cell_choice(x0, x1, *f, N_cells=None, seed=None):
    """Randomly choose from mass-weighted series of k-dimensional hyperboxes.

    Given N k-dimensional hyperboxes with densities known only at 2^k corners
    of each, estimate masses via trapezoid rule then randomly choose from
    mass-weighted list.

    Parameters
    ----------
    x0 : 1D numpy array, length N, or 2D numpy array, shape (N, k) 
        Coordinates of 'first' corner of each hyperbox. 1D array if 1D cells,
        otherwise 2D.
    x1 : 1D numpy array, length N, or 2D numpy array, shape (N, k)
        Coordinates of 'last' corner of each hyperbox. 1D array if 1D cells,
        otherwise 2D.
    *f : 2^k 1D numpy arrays, length N
        Densities at corners of batch of N k-D hypercubes.
    N_cells : int, optional
        Number of cells to sample. Default is None, in which case a single cell
        index (integer) is returned. Otherwise, 1D array of cell indices is
        returned.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    idx : 2D (N_cells x k) or 1D (k,) numpy array of ints
        Indices along each dimension of randomly sampled cells. 2D if N_cells is
        set with an integer (including 1), 1D if N_cells is set to None.
    """
    # check appropriate number of densities given
    if len(f) == 0:
        raise ValueError("Expected corner densities f.")
    if (len(f) & (len(f)-1) != 0):
        raise ValueError("Expected no. corner densities to be power of 2.")
        
    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # mass = average density * volume
    V = np.prod(x1 - x0, axis=-1)
    f_avg = np.average(np.stack(f), axis=0)
    m = f_avg * V

    # normalise mass into probability array
    p = m / m.sum()

    # choose cells
    return rng.choice(len(p), p=p, size=N_cells)
