import numpy as np
from math import log2
from .unitsample_kd import _unitsample_kd
from .utils import _check_N_samples


def sample(x0, x1, *f, N_samples=None, seed=None):
    """Draw sample(s) from k-D hyperbox(es) with known vertex densities.

    Given a k-dimensional hyperbox (or a set of such boxes) with densities known
    only at the 2^k corners of the box(es), draw a sample (or series of samples)
    via interpolant sampling. If a series of boxes, first estimate mass(es) via 
    trapezoid rule then randomly choose from mass-weighted list.
    
    Parameters
    ----------
    x0, x1 : scalar or 1D array_like, length k, or 2D array_like, shape (N, k) 
        Coordinates of 'first' and 'last' cell corners. Scalar if considering
        single 1D cell. 1D array if considering single k-D cell. 2D
        array if considering series of N k-D cells. Note: if considering
        series of 1D cells, need shape (N, 1), not (N).
    *f : 2^k scalars or 2^k 1D array_like, length N.
        Densities at cell corners. Each f should be scalar is considering single
        cell or 1D array if considering series of cells. Densities should all be
        positive.
    N_samples : {None, int}, optional
        Number of samples to draw. Default is None, in which case a single
        sample is drawn.
    seed : {None, int, ``numpy.random.Generator``}, optional
        Seed for ``numpy`` random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See ``numpy`` random generator docs for
        more information.

    Returns
    -------
    X : scalar, 1D array (length k OR N_samples) or 2D array (N_samples, k)
        Sample(s) from linear interpolant. Scalar if single sample (i.e.,
        N_samples is None) in 1D. 1D array if single sample in k-D OR multiple
        samples in 1D. 2D array if multiple samples in k-D.

    Examples
    --------
    By combining input types in various ways, there are essentially 6 ways to
    use ``sample``. We will see each of these in the following examples. In
    each case, we'll just generate densities from a uniform distribution, but
    in general they might come from any arbitrary density function.
    
    1. A single sample from a single 1D cell. Because this is 1D, ``x0`` and
    ``x1`` can be specified as scalars rather than arrays (the latter being the
    case in higher dims), and two densities ``*f`` need to be specified. Here,
    the cell spans ``x0=10`` to ``x1=20``, with corresponding densities 0.1 and
    0.3 respectively.

    >>> sample(10, 20, 0.1, 0.3)
    18.55430897524264
    
    This returns a single scalar, representing a sample between ``x0`` and
    ``x1``.

    2. Multiple samples from a single 1D cell. This looks largely the same as
    the previous example, but now to request multiple samples, one provides an
    integer value to the optional parameter ``N_samples``. We'll use the same
    cell as the previous example.

    >>> sample(10, 20, 0.1, 0.3, N_samples=5)
    array([14.94884745, 19.91094801, 14.5518253 , 17.0042425 , 18.34062011])

    This returns a 1D array, length ``N_samples``.
    
    3. A single sample from a single k-D cell. Because this is k-dimensional, 
    ``x0`` and ``x1`` can no longer be specified as scalars, but should instead
    be length-k vectors (i.e., 1D arrays). The input densities are still
    scalars, but now ``sample`` expects 2^3=8 of them (i.e., the 8 vertices of a
    3D cuboid). Here, we'll take a 3D cell, spanning ``x0 = [10, 100, 1000]`` to
    ``x1=[20, 200, 2000]``, and randomly generated densities.
    
    >>> x0 = np.array([10, 100, 1000])
    >>> x1 = np.array([20, 200, 2000])
    >>> f = np.random.uniform(size=8)
    >>> sample(x0, x1, *f)
    array([  14.44960227,  194.48916323, 1246.63901064])

    This returns a 1D array (length k), representing the single sample point
    within the specified k-D box.

    4. Multiple samples from a single k-D cell. We'll use the same cell as the
    previous example.
    
    >>> x0 = np.array([10, 100, 1000])
    >>> x1 = np.array([20, 200, 2000])
    >>> f = np.random.uniform(size=8)
    >>> sample(x0, x1, *f, N_samples=6)
    array([[  12.63103673,  186.7514952 , 1716.6187807 ],
           [  14.67375968,  116.20984414, 1557.59629547],
           [  11.47055697,  178.41650558, 1592.18260186],
           [  12.41780309,  105.28009531, 1436.39525998],
           [  13.44764381,  152.57623376, 1880.55963378],
           [  18.5522151 ,  133.87092063, 1558.85620176]])

    This returns a 2D array, shape ``N_samples`` x k.

    5. A single sample from a series of N k-D cells. This might represent, for
    example, the set of cells in a k-D grid. Now, ``x0`` and ``x1`` should be
    2D arrays, shape N x k. Note that this includes the case where k=1 (i.e.,
    with a series of N 1D cells, ``x0``, ``x1`` should be shaped N x 1, not N).
    The 2^k densities should each be 1D arrays, length N. Here we'll take 5 2D
    cells, again randomly generating densities.
    
    >>> x0 = np.array([[10, 100], [20, 200], [30, 300], [40, 400], [50, 500]])
    >>> x1 = np.array([[20, 200], [30, 300], [40, 400], [50, 500], [60, 600]])
    >>> f = tuple(np.random.uniform(size=5) for i in range(4))
    >>> sample(x0, x1, *f)
    array([ 26.43027166, 240.69133195])

    This returns a 1D array, length k. This is the single sampling point across
    the k-D sampling space spanned by the series of specified cells.
    
    6. Multiple samples from a series of N k-D cells. We'll use the same set
    of cells as in the previous example.
    
    >>> x0 = np.array([[10, 100], [20, 200], [30, 300], [40, 400], [50, 500]])
    >>> x1 = np.array([[20, 200], [30, 300], [40, 400], [50, 500], [60, 600]])
    >>> f = tuple(np.random.uniform(size=5) for i in range(4))
    >>> sample(x0, x1, *f, N_samples=7)
    array([[ 59.8448691 , 598.90797448],
           [ 11.74636295, 145.18978709],
           [ 30.22930825, 343.08761123],
           [ 40.11901342, 470.0128441 ],
           [ 37.06187148, 304.1873569 ],
           [ 58.81175529, 506.3814689 ],
           [ 37.5928607 , 303.74784732]])

    This returns a 2D array, shape ``N_samples`` x k. This represents the
    ``N_samples`` sampling points, drawn from across the k-D sampling space
    spanned by the series of specified cells.
    """
    # check requested no. samples is None or positive int
    _check_N_samples(N_samples)
    
    # check densities positive everywhere
    for fi in f:
        if np.any(fi < 0):
            raise ValueError("Densities can't be negative")
    
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

    # check x1 > x0 everywhere
    if np.any(x1 <= x0):
        raise ValueError("Need x1 > x0 everywhere")

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
    z = x0 + (x1 - x0) * _unitsample_kd(*f, seed=rng)

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
