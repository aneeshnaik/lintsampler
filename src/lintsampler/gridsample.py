import numpy as np
import warnings
from functools import reduce
from .unitsample_kd import _unitsample_kd
from .utils import _check_N_samples, _generate_usamples, _choice


def _gridsample(
    *edgearrays, f, N_samples=None, seed=None, qmc=False, qmc_engine=None
):
    """Draw sample(s) from density function defined on k-D grid.
    
    Given a k-dimensional grid shaped :math:`(N_0, N_1, ..., N_{k-1})`, the user
    specifies a sequence of k 1D arrays (lengths :math:`N_0+1, N_1+1`, etc.)
    representing the (not necessarily evenly spaced) gridlines along each
    dimension, and a kD array shaped :math:`(N_0+1, N_1+1, ...)` representing
    the (not necessarily normalised) densities at the grid corners. This
    function then draws a sample (or N samples) from this grid. It first chooses
    a cell (or N cells with replacement), weighting them by their mass
    (estimated by the trapezoid rule) then samples from k-linear interpolant
    within the chosen cell(s).

    Parameters
    ----------
    *edgearrays : one or more 1D array_like
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively. The edges do *not* need to be evenly
        spaced.
    f : k-D array_like, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at vertices of k-dimensional grid. Densities
        should all be positive.
    N_samples : {None, int}, optional
        Number of samples to draw. Default is None, in which case a single
        sample is drawn.
    seed : {None, int, ``numpy.random.Generator``}, optional
        Seed for ``numpy`` random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See ``numpy`` random generator docs for
        more information.
    qmc : bool, optional
        Whether to use Quasi-Monte Carlo sampling. Default is False.
    qmc_engine : {None, scipy.stats.qmc.QMCEngine}, optional
        QMC engine to use if qmc flag above is True. Should be subclass of
        scipy QMCEngine, e.g. qmc.Sobol. Should have dimensionality k+1, because
        first k dimensions are used for lintsampling, while last dimension is
        used for cell choice (this happens even if only one cell is given).
        Default is None. In that case, if qmc is True, then a scrambled Sobol
        sequence is used.

    Returns
    -------
    X : scalar, 1D array (length k OR N_samples) or 2D array (N_samples, k)
        Sample(s) from linear interpolant. Scalar if single sample (i.e.,
        N_samples is None) in 1D. 1D array if single sample in k-D OR multiple
        samples in 1D. 2D array if multiple samples in k-D.
    
    Examples
    --------
    
    These examples demonstrate the multiple ways to use ``gridsample``. In
    each case, we'll just generate densities from a uniform distribution, but
    in general they might come from any arbitrary density function.

    1. A single sample from a 1D grid. The grid spans x=0 to x=10, and has 32
    cells (so 33 edges). 
    
    >>> x = np.linspace(0, 10, 33)
    >>> f = np.random.uniform(size=33)
    >>> gridsample(x, f=f)
    0.7355598727871656

    This returns a single scalar: the sampling point within the grid.
    
    2. Multiple samples from a 1D grid (same grid as previous example).
    
    >>> x = np.linspace(0, 10, 33)
    >>> f = np.random.uniform(size=33)
    >>> gridsample(x, f=f, N_samples=4)
    array([0.7432799 , 6.64118763, 9.65968316, 5.39087554])

    This returns a 1D array: the ``N_samples`` sampling points within the grid.

    3. Single sample from a k-D grid. In this case we'll take a 2D grid, with
    32 x 64 cells (so 33 gridlines along one axis and 65 along the other, and
    33x65=2145 intersections with known densities).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> f = np.random.uniform(size=(33, 65))
    >>> gridsample(x, y, f=f)
    array([  7.67294632, 190.45302915])

    This returns a 1D array: the single k-D sampling point within the grid.

    4. Multiple samples from a k-D grid (same grid as previous example).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> f = np.random.uniform(size=(33, 65))
    >>> gridsample(x, y, f=f, N_samples=5)
    array([[1.35963966e-01, 1.38182930e+02],
           [6.52704300e+00, 1.63109912e+02],
           [4.35226761e+00, 1.49753235e+02],
           [3.56093155e+00, 1.48548481e+02],
           [1.31163401e+00, 1.59335676e+02]])

    This returns a 2D array, shape ``N_samples`` x k: the ``N_samples`` k-D
    samples within the grid.
    """
    # check requested no. samples is None or positive int
    _check_N_samples(N_samples)

    # warn if qmc engine provided but qmc off
    if not qmc and qmc_engine is not None:
        warnings.warn("Provided qmc_engine won't be used as qmc switched off.")
    
    # warn if qmc engine provided and RNG seed provided
    if qmc_engine is not None and seed is not None:
        warnings.warn("Provided random seed won't be used as qmc_engine provided.")

    # check edge arrs 1D, mono. increasing, and match corresponding f dim
    for i, a in enumerate(edgearrays):
        if a.ndim != 1:
            raise TypeError("Expected 1D edge arrays.")
        if len(a) != f.shape[i]:
            raise ValueError(
                "Length of edge array doesn't match corresponding "
                "density grid dimension."
            )
        if np.any(np.diff(a) <= 0):
            raise ValueError("Edge array not monotically increasing.")
            
    # check densities positive everywhere
    if np.any(f < 0):
        raise ValueError("Densities can't be negative")

    # check shapes of edge arrays / f make sense
    if f.shape != tuple(len(a) for a in edgearrays):
        raise ValueError("Shape of densities doesn't match edge array lengths.")

    # generate uniform samples (N_samples, k+1) if N_samples, else (1, k+1)
    # first k dims used for lintsampling, last dim used for cell choice
    if N_samples:
        u = _generate_usamples(N_samples, f.ndim + 1, seed, qmc, qmc_engine)
    else:
        u = _generate_usamples(1, f.ndim + 1, seed, qmc, qmc_engine)

    # randomly choose grid cell(s)
    cells = _gridcell_choice(*edgearrays, f=f, u=u[..., -1])
    if not N_samples:
        cells = cells[0]

    # get 2^k-tuple of densities at cell corners
    corners = _gridcell_corners(f, cells)

    # sample on unit hypercube
    z = _unitsample_kd(*corners, u=u[..., :-1])

    # rescale coordinates (loop over dimensions)
    for d in range(f.ndim):
        e = edgearrays[d]
        if N_samples:
            c = cells[:, d]
            z[:, d] = e[c] + np.diff(e)[c] * z[:, d]
        else:
            c = cells[d]
            z[d] = e[c] + np.diff(e)[c] * z[d]

    # squeeze down to scalar / 1D if appropriate
    if not N_samples and (f.ndim == 1):
        z = z.item()
    elif (f.ndim == 1):
        z = z[:, 0]

    return z


def _gridcell_faverages(f):
    """Given grid of densities, evaluated at corners, calculate cell averages.

    Parameters
    ----------
    f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.

    Returns
    -------
    average : k-dim numpy array, shape (N0 x N1 x ... x N{k-1})
        Density in each grid cell, averaging over 2**k corners. Shape is shape
        of grid.
      
    """
    # infer dimensionality and grid shape
    ndim = f.ndim
    shape = tuple([s - 1 for s in f.shape])

    # loop over corners, add density contribution to sum
    # at each corner construct slice tuple, e.g. a[:-1,:-1,etc] for 1st corner
    sum = np.zeros(shape)
    slice0 = slice(-1)
    slice1 = slice(1, None)
    for i in range(2**ndim):
        n = np.binary_repr(i, width=ndim)
        t = ()
        for d in range(ndim):
            t += ([slice0, slice1][int(n[d])],)
        sum += f[t]
    
    # div. by no. corners for average
    average = sum / 2**ndim
    return average


def _gridcell_volumes(*edgearrays):
    """From a sequence of arrays of edge lines, calculate grid cell volumes.
    
    Calculates difference arrays with numpy.diff, then volumes with outer
    product.

    Parameters
    ----------
    *edgearrays : 1 or more 1D numpy arrays
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively.
    
    Returns
    -------
    vols : numpy array, k-dimensional, shape (N0 x N1 x ... x N{k-1})
        k-dimensional array containing volumes of grid cells. Shape is shape of
        grid.
    """
    diffarrays = []
    for edgearray in edgearrays:
        diffarrays.append(np.diff(edgearray))
    shape = tuple([d.size for d in diffarrays])
    vols = reduce(np.outer, diffarrays).reshape(shape)
    return vols


def _gridcell_choice(*edgearrays, f, u):
    """From k-dimensional grid of densities, choose mass-weighted cell(s).

    Given a k-dimensional grid, shaped (N0 x N1 x ... x N{k-1}), the user
    specifies a sequence of k 1D arrays (lengths N0+1, N1+1, etc.) representing
    the (not necessarily evenly spaced) gridlines along each dimension, and a
    kD array (shape N0+1 x N1+1 x ...) representing the densities at the grid
    corners. This function then calculates the mass of each grid cell according
    to the trapezoid rule (i.e., the average density over all 2**k corners times
    the volume of the cell), then randomly chooses a cell (or several cell) from
    the set of cells, weighting each cell by its mass in this choice.

    Parameters
    ----------
    *edgearrays : 1 or more 1D numpy arrays
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively. The edges do *not* need to be evenly
        spaced.
    f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.
    u : 1D numpy array, length N_cells
        Array of uniform samples, length equal to number of desired cells.

    Returns
    -------
    idx : 2D numpy array (N_cells x k)
        Indices along each dimension of randomly sampled cells.
    """
    # calculate cell volumes
    V = _gridcell_volumes(*edgearrays)
    
    # calculate cell average densities
    f_avg = _gridcell_faverages(f)
    
    # mass = density * volume
    m = f_avg * V
    
    # normalise mass and flatten into probability array
    m_norm = m / m.sum()
    p = m_norm.flatten()

    # choose cells
    cells = _choice(p=p, u=u)

    # unravel 1D cell indices into k-D grid indices
    idx = np.stack(np.unravel_index(cells, m_norm.shape), axis=-1)
    return idx


def _gridcell_corners(f, cells):
    """From gridded densities, get densities on 2^k corners of given cells.

    Parameters
    ----------
    f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.
    cells : 2-d numpy array, shape (N, k)
        Grid indices of N chosen cells along the k dimensions of the grid.
        
    Returns
    -------
    corners : 2^k-tuple of 1D numpy arrays, each length N
        Densities at corners of given cells. Conventional ordering applies,
        e.g., in 3D: corners = (f000, f001, f010, f011, f100, f101, f110, f111).

    """
    # infer dimensionality from shape of corner density grid
    ndim = f.ndim
    
    # get loop over 2^k corners, get densities at each
    corners = []
    for i in range(2**ndim):
        
        # binary representation of corner, e.g. [0,0,...,0] is first corner
        n = np.binary_repr(i, width=ndim)
        n = np.array([int(c) for c in n], dtype=int)
 
        # get densities on given corners
        idx = cells + n
        idx = np.split(idx.T, ndim)
        idx = tuple([idxi.squeeze() for idxi in idx])
        corners.append(f[idx])
    return tuple(corners)
